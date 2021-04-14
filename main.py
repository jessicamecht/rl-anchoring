import torch 
import random 
import torch.nn as nn 
import torch.optim as optim
from itertools import count
import math 
from models.models import * 
import csv 
import numpy as np
import matplotlib.pyplot as plt 
from utils import * 

#####CONFIG#########################################
batch_size = 10 
gamma = 0.99
eps_start = 0.9
eps_end = 0.05
eps_decay = 200
####################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''Pseudocode:
1. for each reviewer, sample review session
2. for each review session, sample student 
2.1 send student through NN 
2.1.1 send all past reviewed students through anchoring net and obtain anchoring factor

2.2 obtain action to take weighted by anchoring factor - admission decision with expl-exploit

2.3 get next state - next student (terminal is last student of sequence)
2.4 observe reward and use to update anchor net 
2.5 store transition (state, action, reward, next_state)
2.6 sample last sequences of transitions 
2.7 set y = r if terminal state else r + max Q(state, a, theta) (Bellman equation)
2.8 compute loss and descent
2.9 Update target net 
'''

def main():
    ### Init Data ###################################
    data = load_data()
    n_actions = 2
    _, _, _, data_instance,_ ,_ = data["reviewer_0"][0][-1]
    input_size = len(data_instance)
    keys = np.array(list(data.keys()))
    n_folds = 10
    folds = np.array_split(keys, n_folds) #10-fold cross validation 
    
    for i in range(n_folds-1):
        print("Fold: ", i)
        ### Load Models ###################################
        policy_net = DQN(input_size, n_actions).to(device)
        target_net = DQN(input_size, n_actions).to(device)
        anchoring_net = AnchorNet(input_size+1).to(device)
        anchor_param = nn.Parameter(torch.ones(1), requires_grad=True).to(device).detach().requires_grad_(True)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        ### Init Optimizer and Memory ###################################
        anchor_optimizer = optim.Adam(anchoring_net.parameters(), lr=0.01)
        optimizer = optim.RMSprop(list(policy_net.parameters()))# + [anchor_param])
        memory = ReplayMemory(10000)

        first_part = [item for sublist in folds[0:i] for item in sublist] 
        second_part = [item for sublist in folds[i+2:] for item in sublist] if len(folds) > i+2 else []
        train_keys = first_part + second_part 
        valid_keys = folds[i] 
        test_keys = folds[i+1]
        steps_done = 0
    
        _ = train(data, train_keys, valid_keys, steps_done, eps_end, eps_start, eps_decay, policy_net, target_net, anchoring_net, optimizer, anchor_optimizer, anchor_param, memory)
        
        validate(data, valid_keys, target_net)
        validate(data, test_keys, target_net, label="test")


def select_action(steps_done, eps_end, eps_start, eps_decay, policy_net, state, anchor, n_actions=2):
    '''selects the next action to be executed based on the current student shown
    actions can be 0: rejection, 1:admission
    '''
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done/eps_decay)
    #exploration/exploitation trade-off
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            output = policy_net(state) #
            return steps_done + 1, torch.tensor(output.max(1)[1].view(1, 1) * anchor, dtype=torch.int64, device=device)
    else:
        return steps_done + 1, torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model(batch_size, memory, policy_net, target_net):
    '''optimizes the model by taking a batch of experiences from memory and computing the
    loss between the expected q values and the current q values with L1 loss'''

    batch_size = min(len(memory.memory), batch_size)

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])


    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    output = policy_net(state_batch)
    state_action_values = output.gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch


    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    loss.backward()

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    return loss 

def reward(action, target_decision):
    '''calculates the reward for the current decision'''
    action = action.item()
    target_decision = int(target_decision)
    return int(action==target_decision)

def anchor_loss(anchor, prediction, target):
    action = torch.tensor(prediction.max(1)[1].view(1, 1) * anchor, dtype=torch.int64, device=device)

    loss = -1* (target-anchor*action)
    return loss 

def validate(data, valid_keys, target_net, label="validation"):
    '''sends unseen data into the target net and observes the average validation reward on this data'''
    target_net.eval()
    cum_reward = 0
    num_reviews = 0 
    for reviewer in data:
        if reviewer not in valid_keys:
            continue
        for review_session in data[reviewer]:
            for idx, student in enumerate(review_session):
                
                timestamp, target_decision, final_decision, features, svm_decision, svm_confidence = student
                target_decision = int(target_decision>1)
                features = torch.Tensor(features).to(device).unsqueeze(0)

                output = target_net(features)
                action = output.max(1)[1].view(1, 1)
                cum_reward += torch.tensor([reward(action, target_decision)], device=device).item()
                num_reviews+=1
    print("average ", label, " reward: ", cum_reward/num_reviews)
    

def train(data, train_keys, valid_keys, steps_done, eps_end, eps_start, eps_decay, policy_net, target_net, anchoring_net, optimizer, anchor_optimizer, anchor_param,  memory):
    '''main training function 
    iterates over all reviewers and each review session 
    for each student of the review session, 
    '''
    losses_all = []
    num_episodes = 1
    for i_episode in range(num_episodes):
        missing_decisions = 0
        losses = []
        all_rewards = 0
        all_reviews = 0
        for reviewer in data:
            cum_reward = 0
            number_reviews = 0
            if reviewer not in train_keys:
                continue
            anchors = []
            for review_session in data[reviewer]:
                past_students = []
                for idx, student in enumerate(review_session):

                    timestamp, target_decision, final_decision, features, svm_decision, svm_confodence = student
                    if target_decision is None or final_decision is None:
                        continue

                    target_decision = int(target_decision>1)
                    past_students.append(features + [target_decision])

                    features = torch.Tensor(features).to(device).unsqueeze(0)

                    past_reviewed_students = torch.tensor(past_students, device=device)
                    #anchor = anchoring_net(past_reviewed_students)
                    anchor = torch.ones(1).to(device)#anchor_param

                    steps_done, action = select_action(steps_done, eps_end, eps_start, eps_decay, policy_net, features, anchor)

                    curr_reward = torch.tensor([reward(action, final_decision)], device=device)
                    cum_reward+=curr_reward.item()
                    number_reviews+=1
                    
                    next_student = review_session[idx+1] if idx+1<len(review_session) else (_, _, _, None, _,_)
                    _, _, _, next_student, _, _ = next_student
                    next_student = torch.Tensor(next_student).to(device).unsqueeze(0) if next_student != None else None

                    memory.push(features, action, next_student, curr_reward)

                    optimizer.zero_grad()
                    loss = optimize_model(batch_size, memory, policy_net, target_net)
                    losses.append(loss.item())
                    optimizer.step()
                    anchors.append(anchor.item())

                    '''with torch.no_grad():
                        prediction = policy_net(features)
                    anchor_optimizer.zero_grad()
                    anchor_loss_value = anchor_loss(anchor, prediction, target_decision)
                    anchor_loss_value.backward()
                    anchor_optimizer.step()'''
            #print('anchors for review session', anchors)

            all_rewards+=cum_reward
            all_reviews+=number_reviews
            #print('Average reward for ', reviewer, ": ", cum_reward/number_reviews, "who reviewed ", number_reviews, " students for episode: ", i_episode)
        print('average train reward: ', all_rewards/all_reviews)
        #validate(data, valid_keys, target_net)

        target_net.load_state_dict(policy_net.state_dict())
        losses_all.append(losses)
        #print(missing_decisions, " missing decisions for ", steps_done, " steps done")
    return losses_all

if __name__ == "__main__":
    main()
    





