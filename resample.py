import torch 
import numpy as np 
from utils import * 
import random 
from action_selection import * 
from models.actor_critic_models import * 
import torch.optim as optim
from models.models import AnchorLSTM
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eps_start = 0.9
eps_end = 0.05
eps_decay = 200


def heuristic_resample(data, anchor_lstm, keys):
    '''input pool of students per year'''
    '''Pseudocode: 
    1. create fictional reviewer who reviews a random number of x students where x > 1 and x < 30
    2. sample one student
    3. run it through svm
    4. run through lstm to obtain anchor 
    5. sample next student based on anchor 
    6. run sequence through lstm and obtain anchor 
    7. evauate the average anchor 
    '''
    all_students = students_by_year(data, keys)
    anchor_lstm.eval()

    sum_bias = 0 
    all_decisions = 0
    for student_pool_for_year in all_students:
        while len(student_pool_for_year) > 0:
            length_of_sequence = min(random.randint(2,30), len(student_pool_for_year))
            student_sequence = []
            last_anchor = torch.zeros(1) + 0.5
            anchor = 0
            for i in range(length_of_sequence):

                remove_idx, student = heuristic_select_next_action(last_anchor, student_pool_for_year)
                student_sequence.append(student)
                student_pool_for_year = np.delete(student_pool_for_year, remove_idx, axis=0)
                
                hidden_size = 1
                timestamp, target_decision, final_decision, features, svm_decision, svm_confidence = student
                hidden_anchor_states = (torch.zeros(1,1,hidden_size).to(device),
                            torch.zeros(1,1,hidden_size).to(device))
                svm_decision = np.array(student_sequence)[:,-2]
                svm_decision = torch.Tensor(np.array(svm_decision, dtype=int))


                svm_decision = torch.unsqueeze(torch.unsqueeze(svm_decision, 0), -1).to(device)
                anchor, _ = anchor_lstm(svm_decision,hidden_anchor_states)

                last_anchor = anchor.squeeze(0)[-1]

                if i == length_of_sequence-1:
                    
                    norm_anch = normalize(anchor)
                    
                    if torch.isnan(torch.abs(norm_anch).sum()).any():
                        print(norm_anch, anchor)
                    sum_bias+= torch.abs(norm_anch).sum()
                    all_decisions+=anchor.shape[1]

    print("Heuristic Resampled Average Absolute Anchor: ", (sum_bias/all_decisions).item())

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def anchor_reward(anchors):
    return (1 - torch.abs(anchors)).sum() # 1 if anchors are 0, 0 if anchors are 1,-1

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

def train_learned_resampling_Q(data, keys, n_iters, anchor_lstm):
    '''Pseudocode
    actor resamples a new student 
    critic critics the student that was sampled
    
    '''

    all_students = all_students_sorted_by_year(data, keys)
    anchor_lstm.eval()

    state_size = 1
    input_size = 1
    n_actions = len(all_students)
    policy_net = DQN(input_size, n_actions).to(device)
    target_net = DQN(input_size, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    ### Init Optimizer and Memory ###################################
    optimizer = optim.RMSprop(list(policy_net.parameters()))# + [anchor_param])
    memory = ReplayMemory(10000)

    for iteration in range(n_iters):

        sum_bias = 0 
        all_decisions = 0
        all_students_mask = np.array(list(range(len(all_students[:,0])))) != -1
        steps_done=0

        for year in range(max(all_students[:,0])):
                length_of_sequence = min(random.randint(2,30), len(all_students))
                student_sequence = []
                hidden_size=1
                hidden_anchor_state = (torch.zeros(1,1,hidden_size).to(device), torch.zeros(1,1,hidden_size).to(device))#initial anchor 
                #sample random student 
                all_students_mask = np.logical_and(np.array(all_students_mask), (np.array(all_students[:,1]) == year))
                possible_students = all_students[all_students_mask]
                if len(possible_students) == 0:
                    break
                student = possible_students[0]
                student_sequence.append(student)
                for i in range(length_of_sequence):
                    steps_done+=1
                    svm_decision = np.array(student_sequence)[:,-2]
                    svm_decision = torch.Tensor(np.array(svm_decision, dtype=int))
                    svm_decision = torch.unsqueeze(torch.unsqueeze(svm_decision, 0), -1).to(device)

                    with torch.no_grad():
                        state, _ = anchor_lstm(svm_decision,hidden_anchor_state)

                    output = policy_net(state)

                    #eliminate impossible actions (already sampled students and students from a different year)
                    valid_output = output * torch.tensor(all_students_mask).to(device)
                    #Select student to be sampled
                    action_idx = select_action(valid_output,eps_end, eps_start, eps_decay, steps_done).item()
                    next_student = all_students[action_idx]


                    reward = anchor_reward(state)

def train_learned_resampling(data, keys, keys_valid, n_iters, anchor_lstm):
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
        sum_bias = 0 
        all_decisions = 0
        all_students_mask = np.array(list(range(len(all_students_train[:,0])))) != -1
        all_students_mask_valid = np.array(list(range(len(all_students_val[:,0])))) == -1
        all_students_mask = np.concatenate((all_students_mask,all_students_mask_valid))
        steps_done=0

        for year in range(max(all_students[:,0])):
                length_of_sequence = min(random.randint(2,30), len(all_students))
                student_sequence = []
                hidden_size=1
                hidden_anchor_state = (torch.zeros(1,1,hidden_size).to(device), torch.zeros(1,1,hidden_size).to(device))#initial anchor 
                #sample random student 
                all_students_mask = np.logical_and(np.array(all_students_mask), (np.array(all_students[:,1]) == year))
                possible_students = all_students[all_students_mask]

                if len(possible_students) == 0:
                    break

                student = possible_students[0]
                student_sequence.append(student)

                log_probs = []
                values = []
                rewards = []
                masks = []
                entropy = 0
                for i in range(length_of_sequence):
                    steps_done+=1
                    svm_decision = np.array(student_sequence)[:,-2]
                    svm_decision = torch.Tensor(np.array(svm_decision, dtype=int))
                    svm_decision = torch.unsqueeze(torch.unsqueeze(svm_decision, 0), -1).to(device)

                    with torch.no_grad():
                        state, _ = anchor_lstm(svm_decision,hidden_anchor_state)

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
                
                sum_bias += torch.abs(normalize(state)).sum()
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

            
                print("Training Average Absolute Anchor: ", sum_bias/all_decisions)

                optimizerA.zero_grad()
                optimizerC.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                optimizerA.step()
                optimizerC.step()
    return actor

def eval_learned_resampling(data, keys, keys_valid, anchor_lstm, actor):
    all_students = all_students_sorted_by_year(data, keys)
    val_students = all_students_sorted_by_year(data, keys_valid)
    anchor_lstm.eval()
    actor.eval()

    sum_bias = 0 
    all_decisions = 0
    all_students_mask = np.array(list(range(len(all_students[:,0])))) != -1
    val_students_mask = np.array(list(range(len(val_students[:,0])))) == -1
    all_students_mask = np.concatenate((all_students_mask, val_students_mask))

    all_students = np.concatenate((all_students, val_students))
    steps_done = 200

    for year in range(max(all_students[:,0])):
        length_of_sequence = min(random.randint(2,30), len(all_students))
        student_sequence = []
        hidden_size=1
        hidden_anchor_state = (torch.zeros(1,1,hidden_size).to(device), torch.zeros(1,1,hidden_size).to(device))#initial anchor 
        #sample random student 
        all_students_mask = np.logical_and(np.array(all_students_mask), (np.array(all_students[:,1]) == year))
        possible_students = all_students[all_students_mask]

        if len(possible_students) == 0:
            break

        student = possible_students[0]
        student_sequence.append(student)

        for i in range(length_of_sequence):
            svm_decision = np.array(student_sequence)[:,-2]
            svm_decision = torch.Tensor(np.array(svm_decision, dtype=int))
            svm_decision = torch.unsqueeze(torch.unsqueeze(svm_decision, 0), -1).to(device)

            with torch.no_grad():
                state, _ = anchor_lstm(svm_decision,hidden_anchor_state)

            #get probability for action and critic given anchor 
            output = actor(state[:,-1:,:])#only feed the last state 

            #eliminate impossible actions (already sampled students and students from a different year)
            valid_output = output * torch.tensor(all_students_mask).to(device)
            #Select student to be sampled
            action_idx = select_action(valid_output,eps_end, eps_start, eps_decay, steps_done).item()
            next_student = all_students[action_idx]

            student_sequence.append(next_student)
            #remove the student s.t. he can't be sampled again  
            all_students_mask[action_idx] = False

        sum_bias += torch.abs(normalize(state)).sum()
        all_decisions +=  len(state.squeeze(0))
    print("Resampled Average Absolute Anchor: ", (sum_bias/all_decisions).item())

def main():
    data = load_data()
    _, _, _, data_instance, _, _ = data["reviewer_0"][0][-1]
    action_size = len(data_instance)
    keys = np.array(list(data.keys()))
    n_folds = 10
    folds = np.array_split(keys, n_folds) #10-fold cross validation 
    hidden_size = 1
    input_size = 1
    anchor_lstm = AnchorLSTM(input_size, hidden_size).to(device)
    anchor_lstm.load_state_dict(torch.load('./state_dicts/anchor_lstm.pt'))
    n_iters=1
    
    for i in range(n_folds-1):
        print("Fold: ", i)
        train_keys = [item for sublist in folds[0:i] for item in sublist]  + [item for sublist in folds[i+2:] for item in sublist] if len(folds) > i+2 else [] 
        valid_keys = folds[i] 
        test_keys = folds[i+1]
        if train_keys != []:
            actor = train_learned_resampling(data, train_keys, valid_keys, n_iters, anchor_lstm)
            eval_learned_resampling(data, train_keys, valid_keys, anchor_lstm, actor)
            heuristic_resample(data, anchor_lstm, valid_keys)
        #validate()

if __name__ == "__main__":
    main()
