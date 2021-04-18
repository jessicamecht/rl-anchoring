import torch.nn as nn 
import torch
import random 
from collections import namedtuple
import torch.nn.functional as F


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))#a named tuple representing a single transition in our environment. maps (state, action) pairs to their (next_state, reward) result

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory(object):
    '''a cyclic buffer of bounded size that holds the transitions observed recently'''
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        '''saves a transition'''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position+1)%self.capacity #what does this do?

    def sample(self, batch_size):
        '''method for selecting a random batch of transitions for training'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,output_size)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

class AnchorNet(nn.Module):
    def __init__(self, input_size):
        super(AnchorNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,1)
        
    def forward(self,input_x):
        '''anchor net takes in a batch of student features 
        and past decisions and learns an anchoring score'''
        x = self.activation(self.fc1(input_x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        # negative anchoring_factor means negative anchoring bias (has seen lots of bad students)
        # positive anchoring_factor means positive anchoring bias (has seen lots of good students)

        x = x+1
        anchoring_factor = x.sum()/input_x.shape[0]
        return anchoring_factor

class AnchorLSTM(nn.Module):
    '''AnchorLSTM takes in a sequence of ratings of students for the current reviewer and is 
    supposed to learn the anchor in the current hidden state'''
    def __init__(self, input_size, hidden_size, output_size=2):
        super(AnchorLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size= input_size, hidden_size=hidden_size, batch_first=True)

        self.linear = nn.Linear(hidden_size, output_size)

    
    def forward(self, x, h):
        predictions, h = self.lstm(x, h)
        predictions = self.linear(predictions)
        return predictions, h
