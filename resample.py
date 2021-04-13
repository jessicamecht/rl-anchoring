import torch 
import numpy as np 
from utils import * 
import random 
from action_selection import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    print("Resampled Average Absolute Anchor: ", (sum_bias/all_decisions).item())
