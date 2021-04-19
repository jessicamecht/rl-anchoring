import torch 
import random 
import torch.nn as nn 
import torch.optim as optim
from itertools import count
import math 
from models.models import * 
import csv 
from utils import * 
import numpy as np
import matplotlib.pyplot as plt 
import operator
import random 
import pandas as pd
from resample import * 

#####CONFIG#########################################
batch_size = 10 
gamma = 0.99
eps_start = 0.9
eps_end = 0.05
eps_decay = 200
####################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''Pseudocode:
1. send student features through svm to obtain score (already done)
2. send student score sequences through lstm concatenated with 
3. the hidden states are the anchor 
'''

def main(input_for_lstm="SVM"):
    print("Evaluating input for LSTM:", input_for_lstm)
    ### Init Data ###################################
    data = load_data()
    _, _, _, data_instance, _, _ = data["reviewer_0"][0][-1]
    keys = np.array(list(data.keys()))
    folds = np.array_split(keys, 7) #10-fold cross validation 

    accuracy_train = []
    accuracy_val = []
    for i in range(len(folds)):
        print("Fold: ", i)
        ### Load Models ###################################
        input_size = 2 if input_for_lstm == "SVM+Decision" else 1
        hidden_size = 1
        anchor_lstm = AnchorLSTM(input_size, hidden_size).to(device)
        loss_fn = nn.CrossEntropyLoss()

        ### Init Optimizer ###################################
        anchor_optimizer = optim.Adam(anchor_lstm.parameters(), lr=0.01)

        ### Train and Valid Keys ###################################
        train_keys = [item for sublist in folds[0:i] for item in sublist]  + [item for sublist in folds[i+2:] for item in sublist] if len(folds) > i+2 else [] 
        valid_keys = folds[i] 
        test_keys = folds[i+1]
        if train_keys != []:
    
            acc_train = train_anchor(data, train_keys, anchor_lstm, anchor_optimizer, loss_fn, input_for_lstm)
            acc_val = eval_anchor(data, valid_keys, anchor_lstm, input_for_lstm)

            accuracy_train.append(acc_train)
            accuracy_val.append(acc_val)
            torch.save(anchor_lstm.state_dict(), f'./state_dicts/anchor_lstm_{input_for_lstm}.pt')
            print(f'Training accurcy for {i} fold:', acc_train)
            print(f'Validation accurcy for {i} fold:', acc_val)
    
    print(f'Average training accurcy for {n_folds-2} folds:', np.array(accuracy_train).mean())
    print(f'Average validation accurcy for {n_folds-2} folds:', np.array(accuracy_val).mean())

def train_anchor(data, train_keys, anchor_lstm, anchor_optimizer, loss_fn, input_for_lstm):
    '''main training function 
    iterates over all reviewers and each review session 
    for each student of the review session, 
    '''
    anchor_lstm.train()
    num_epochs = 10
    
    for epoch in range(num_epochs):
        num_decisions = 0
        num_correct = 0
        for reviewer in data:
            cum_reward = 0
            number_reviews = 0
            if reviewer not in train_keys:
                continue
            
            for review_session in data[reviewer]:
                hidden_anchor_states = (torch.zeros(1,1,1).to(device),
                            torch.zeros(1,1,1).to(device)) 
                if len(review_session) == 1:
                    continue
                anchor_lstm.zero_grad()

                lstm_input, reviewer_decision = get_input_output_data(review_session, input_for_lstm)
                preds, hidden = anchor_lstm(lstm_input,hidden_anchor_states)

                preds = preds.squeeze(0).to(device)
                loss_ll = loss_fn(preds, reviewer_decision)
                loss_ll.backward()
                anchor_optimizer.step()

                ### Accuracy ##########################################
                decisions = torch.argmax(preds, dim=1) == reviewer_decision
                correct = decisions.sum().item()
                num_decisions+= len(decisions)
                num_correct += correct
    return num_correct/num_decisions

def eval_anchor(data, eval_keys, anchor_lstm, input_for_lstm):
    anchor_lstm.eval()
    num_decisions = 0
    num_correct = 0
    for reviewer in data:
        cum_reward = 0
        number_reviews = 0
        if reviewer not in eval_keys:
            continue            
        for review_session in data[reviewer]:
            hidden_anchor_states = (torch.zeros(1,1,1).to(device),
                            torch.zeros(1,1,1).to(device))
            if len(review_session) == 1:
                continue

            lstm_input, reviewer_decision = get_input_output_data(review_session, input_for_lstm)
            preds, _ = anchor_lstm(lstm_input,hidden_anchor_states)

            preds = preds.squeeze(0).to(device)
            ### Accuracy ##########################################
            decisions = torch.argmax(preds, dim=1) == reviewer_decision
            correct = decisions.sum().item()
            all_decisions = len(decisions)
            num_decisions+= all_decisions
            num_correct += correct
    return num_correct/num_decisions 


if __name__ == "__main__":
    main(input_for_lstm="Decision")
    main(input_for_lstm="SVM+Decision")
    main(input_for_lstm="SVM")
    





