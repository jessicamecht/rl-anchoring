import torch 
import numpy as np 
from utils import * 
import random 
from action_selection import * 
from models.actor_critic_models import * 
import torch.optim as optim
from models.models import AnchorLSTM
import math
import matplotlib.pyplot as plt
import collections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eps_start = 0.9
eps_end = 0.05
eps_decay = 200

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def anchor_reward(anchors):
    return (1 - torch.abs(anchors)).sum()

def select_action(output, eps_end, eps_start, eps_decay, steps_done):
    '''selects the next action to be executed based on the current student shown
    actions can be 0: rejection, 1:admission
    '''
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done/eps_decay)
    #exploration/exploitation trade-off
    if sample > eps_threshold:
        with torch.no_grad():
            return torch.tensor(output.max(2)[1].view(1), dtype=torch.int64, device=device)
    else:
        return torch.tensor([[random.randrange(len(output))]], device=device, dtype=torch.long)

def train_learned_resampling(data, keys, keys_valid, n_iters, anchor_lstm, input_for_lstm, fold):
    '''Pseudocode
    actor resamples a new student 
    critic critics the student that was sampled
    gets reward if bias is low 
    updates the networks
    '''

    all_students_train = all_students_sorted_by_year(data, keys)
    all_students_val = all_students_sorted_by_year(data, keys_valid)
    all_students = np.concatenate((all_students_train, all_students_val))
    anchor_lstm.eval()

    state_size = 1

    actor = Actor(state_size, len(all_students)).to(device)
    critic = Critic(state_size).to(device)
    actor.train()
    critic.train()
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())


    for iteration in range(n_iters):
        all_decisions = 0
        steps_done=0

        review_sessions = []
        for year in range(max(all_students[:,0])):

            all_students_mask = get_student_mask(all_students_train, all_students_val)
            all_students_mask = np.logical_and(np.array(all_students_mask), (np.array(all_students[:,0]) == year))
            possible_students = all_students[all_students_mask]

            while len(possible_students) > 0:

                hidden_anchor_state = (torch.zeros(1,1,1).to(device), torch.zeros(1,1,1).to(device))#initial anchor 
                length_of_sequence = min(random.randint(2,30), len(all_students))
                student_sequence = []
                student = possible_students[0]
                student_sequence.append(student)

                log_probs = []
                values = []
                rewards = []
                masks = []
                entropy = 0
                for i in range(length_of_sequence):
                    steps_done+=1
                    lstm_input, reviewer_decision = get_input_output_data(student_sequence, input_for_lstm)

                    with torch.no_grad():
                        predictions, (state, _) = anchor_lstm(lstm_input,hidden_anchor_state)
                    #get probability for action and critic given anchor 
                    output, value = actor(state[:,-1:,:]), critic(state[:,-1:,:])#only feed the last state 

                    #eliminate impossible actions (already sampled students and students from a different year)
                    valid_output = output * torch.tensor(all_students_mask).to(device)
                    #Select student to be sampled
                    action_idx = select_action(valid_output,eps_end, eps_start, eps_decay, steps_done).item()
                    next_student = all_students[action_idx]

                    student_sequence.append(next_student)
                    #remove the student s.t. he can't be sampled again  
                    all_students_mask[action_idx] = False
                    possible_students = all_students[all_students_mask]


                    # get reward (min sum of anchor)
                    reward = anchor_reward(state)
                    rewards.append(reward)
                    done = all_students_mask.sum()==0
                    log_prob = torch.log(output.squeeze()[action_idx])
                    logp = torch.log2(output.squeeze())
                    entropy = (-output.squeeze()*logp).sum()
                    entropy += entropy.mean()
                    
                    log_probs.append(log_prob.unsqueeze(0))
                    values.append(value.squeeze(2))
                    masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
                review_sessions.append(student_sequence)
                all_decisions +=  len(state.squeeze(0))
                next_state = state
                next_value = critic(next_state)
                
                returns = compute_returns(next_value, rewards, masks)
                log_probs = torch.cat(log_probs)
                returns = torch.cat(returns).detach()
                values = torch.cat(values)
                
                advantage = returns - values

                actor_loss = -(log_probs * advantage.detach()).mean()
                critic_loss = advantage.pow(2).mean()

                optimizerA.zero_grad()
                optimizerC.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                optimizerA.step()
                optimizerC.step()

    correlation(review_sessions)
    plot_n_decisions_vs_confidence(review_sessions, f"./figures/resampled_confidence_at_end_of_training_{input_for_lstm}_{fold}.png")
    return actor

def get_student_mask(all_students_train, all_students_val, val=False):
    if val == False:
        all_students_mask = np.array(list(range(len(all_students_train[:,0])))) != -1
        all_students_mask_valid = np.array(list(range(len(all_students_val[:,0])))) == -1
    else:
        all_students_mask = np.array(list(range(len(all_students_train[:,0])))) == -1
        all_students_mask_valid = np.array(list(range(len(all_students_val[:,0])))) != -1
    all_students_mask = np.concatenate((all_students_mask,all_students_mask_valid))
    return all_students_mask

def eval_learned_resampling(data, keys, keys_valid, anchor_lstm, actor, input_for_lstm, fold):
    all_students_train = all_students_sorted_by_year(data, keys)
    all_students_val = all_students_sorted_by_year(data, keys_valid)
    all_students = np.concatenate((all_students_train, all_students_val))

    anchor_lstm.eval()
    actor.eval()

    all_decisions = 0
    steps_done = 200

    review_sessions = []

    for year in range(max(all_students[:,0])):

            all_students_mask = get_student_mask(all_students_train, all_students_val, val=True)
            all_students_mask = np.logical_and(np.array(all_students_mask), (np.array(all_students[:,0]) == year))
            possible_students = all_students[all_students_mask]

            while len(possible_students) > 0:
                hidden_anchor_state = (torch.zeros(1,1,1).to(device), torch.zeros(1,1,1).to(device))#initial anchor 
                length_of_sequence = min(random.randint(2,30), len(all_students))
                student_sequence = []
                student = possible_students[0]
                student_sequence.append(student)

                for i in range(length_of_sequence):
                    steps_done+=1
                    lstm_input, reviewer_decision = get_input_output_data(student_sequence, input_for_lstm)

                    with torch.no_grad():
                        predictions, (state, _) = anchor_lstm(lstm_input,hidden_anchor_state)
                    #get probability for action and critic given anchor 
                    output = actor(state[:,-1:,:])
                    #eliminate impossible actions (already sampled students and students from a different year)
                    valid_output = output * torch.tensor(all_students_mask).to(device)
                    #Select student to be sampled
                    action_idx = select_action(valid_output,eps_end, eps_start, eps_decay, steps_done).item()
                    next_student = all_students[action_idx]

                    student_sequence.append(next_student)
                    #remove the student s.t. he can't be sampled again  
                    all_students_mask[action_idx] = False
                    possible_students = all_students[all_students_mask]

                review_sessions.append(student_sequence)
                all_decisions +=  len(state.squeeze(0))
    correlation(review_sessions)
    plot_n_decisions_vs_confidence(review_sessions, figname=f'./figures/validation_confidence_{input_for_lstm}_{fold}.png')

def main(input_for_lstm, n_iters=1):
    ### Load data #######################
    data = load_data()
    _, _, _, data_instance, _, _ = data["reviewer_0"][0][-1]
    action_size = len(data_instance)
    keys = np.array(list(data.keys()))
    
    input_size, hidden_size = 2 if input_for_lstm == "SVM+Decision" else 1, 1
    anchor_lstm = AnchorLSTM(input_size, hidden_size).to(device)
    anchor_lstm.load_state_dict(torch.load(f'./state_dicts/anchor_lstm_{input_for_lstm}.pt'))
    
    folds = np.array_split(keys, 7) #10-fold cross validation 
    for i in range(len(folds)):
        print("Fold: ", i)
        train_keys_1 = [item for sublist in folds[0:i] for item in sublist] if i > 0 else []  
        train_keys_2 = [item for sublist in folds[i+1:] for item in sublist] if len(folds) > i+1 else [] 
        train_keys = train_keys_2 + train_keys_1
        valid_keys = folds[i] 

        actor = train_learned_resampling(data, train_keys, valid_keys, n_iters, anchor_lstm, input_for_lstm, i)
        eval_learned_resampling(data, train_keys, valid_keys, anchor_lstm, actor, input_for_lstm, i)

if __name__ == "__main__":
    main("SVM+Decision")
    main("SVM")
    main("Decision")
