import torch 
import random 
import torch.nn as nn 
import torch.optim as optim
from itertools import count
import math 
from models import * 
import csv 
import numpy as np
#####CONFIG#########################################

batch_size = 10 
gamma = 0.99
eps_start = 0.9
eps_end = 0.05
eps_decay = 200
target_update = 10

####################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''Pseudocode:
1. for each reviewer, sample review session
2. for each review session, sample student 
2.1 send student through NN 
2.2 obtain action to take - admission decision with expl-exploit
2.3 get next state - next student (terminal is last student of sequence)
2.4 observe reward
2.5 store transition (state, action, reward, next_state)
2.6 sample last sequence of transitions 
2.7 set y = r if terminal state else r + max Q(state, a, theta) (Bellman equation)

'''
def select_action(steps_done, eps_end, eps_start, eps_decay, policy_net, state, n_actions=2):
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
            return steps_done + 1, output.max(1)[1].view(1, 1)
    else:
        return steps_done + 1, torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model(batch_size, memory, policy_net, target_net):
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
    
def load_data():
    read_dictionary = np.load('../admissions.npy',allow_pickle='TRUE').item()
    return read_dictionary

def main():
    data = load_data()
    n_actions = 2
    _, _, _, data_instance = data["reviewer_0"][0][-1]
    input_size = len(data_instance)
    policy_net = DQN(input_size, n_actions).to(device)
    target_net = DQN(input_size, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)

    steps_done = 0
    train(data, steps_done, eps_end, eps_start, eps_decay, policy_net, target_net, optimizer, memory)

def reward(action, target_decision):
    action = action.item()
    target_decision = int(target_decision)
    return int(action==target_decision)
    
    

def train(data, steps_done, eps_end, eps_start, eps_decay, policy_net, target_net, optimizer, memory):
    for reviewer in data:
        for review_session in data[reviewer]:
            if len(review_session)<=1:#do nothing when only one student was reviewed in a session 
                continue
            for idx, student in enumerate(review_session):
                timestamp, target_grade, target_decision, features = student
                if target_decision is None:
                    continue
                features = torch.Tensor(features).to(device).unsqueeze(0)
                steps_done, action = select_action(steps_done, eps_end, eps_start, eps_decay, policy_net, features)
                curr_reward = torch.tensor([reward(action, target_decision)], device=device)
                next_student = review_session[idx+1] if idx+1<len(review_session) else (_, _, _, None)
                _, _, _, next_student = next_student
                next_student = torch.Tensor(next_student).to(device).unsqueeze(0) if next_student != None else None
                memory.push(features, action, next_student, curr_reward)
        
                optimizer.zero_grad()
                optimize_model(batch_size, memory, policy_net, target_net)
                optimizer.step()
    target_net.load_state_dict(policy_net.state_dict())

if __name__ == "__main__":
    main()
    





