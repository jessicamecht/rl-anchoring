import torch.nn as nn 
import torch
import random 
from collections import namedtuple
import torch.nn.functional as F


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))#a named tuple representing a single transition in our environment. maps (state, action) pairs to their (next_state, reward) result


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
        self.fc1 = nn.Linear(input_size, 16)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8,4)
        self.fc4 = nn.Linear(4,output_size)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x