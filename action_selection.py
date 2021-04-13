import torch 
import numpy as np
from torch.distributions import Categorical

def learned_select_next_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()

def heuristic_select_next_action(anchor, student_pool):
    '''selects a student based on the anchoring score obtained buy the LSTM'''
    rejects_mask = np.array(student_pool)[:,-2] == 0 # select based on SVM score 
    admit_mask = np.array(student_pool)[:,-2] == 1 
    reject = np.array(list(enumerate(student_pool)))[rejects_mask]
    admit = np.array(list(enumerate(student_pool)))[admit_mask]
    sorted_student_pool_reject = sorted(reject,  key = lambda x : x[1][-1])#sort by confidence
    sorted_student_pool_admit = sorted(admit,  key = lambda x : x[1][-1])

    #print(len(sorted_student_pool_admit), "students to admit.", len(sorted_student_pool_reject), "students to reject")
    if len(sorted_student_pool_admit) == 0:
        idx, student =  sorted_student_pool_reject[0]
    elif len(sorted_student_pool_reject) == 0:
        idx, student =  sorted_student_pool_admit[0]
    elif anchor > 0.75:# is biased because of recent admission
        #select a very bad student from student pool -> rejection with high confidence 
        idx, student =  sorted_student_pool_reject[0]
    elif anchor > 0.5: # is just a little biased because of recent admission
        #select an edge case student from student_pool to admit 
        idx, student = sorted_student_pool_reject[-1]
    elif anchor <= 0.5:# is biased because of recent rejection
        #select a very good student from student pool -> admission with high confidence 
        idx, student = sorted_student_pool_admit[0]
    elif anchor <= 0.25 :# is just a little biased because of recent rejection
        #select an edge case student from student_pool to reject 
        idx, student = sorted_student_pool_admit[-1]
   
    return idx, student 