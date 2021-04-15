import torch 
import numpy as np
from torch.distributions import Categorical

def sort_by_confidence(student_pool):
    rejects_mask = np.array(student_pool)[:,-2] == 0 # select based on SVM score 
    admit_mask = np.array(student_pool)[:,-2] == 1 
    reject = np.array(list(enumerate(student_pool)))[rejects_mask]
    admit = np.array(list(enumerate(student_pool)))[admit_mask]
    sorted_student_pool_reject = sorted(reject,  key = lambda x : x[1][-1])#sort by confidence
    sorted_student_pool_admit = sorted(admit,  key = lambda x : x[1][-1])
    return sorted_student_pool_reject, sorted_student_pool_admit

def heuristic_select_next_action(anchor, student_pool):
    '''selects a student based on the anchoring score obtained buy the LSTM'''
    sorted_student_pool_reject, sorted_student_pool_admit = sort_by_confidence(student_pool)

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