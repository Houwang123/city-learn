import torch
import torch.nn as nn
import torch.nn.functional as F
import random, math
import typing
import numpy as np

#TODO: How do we alter centralised critics to deal with different number of buildings?

class CentralCritic(nn.Module):
    '''
    Implements a centralised critic 
    with a multi layer perceptron
    '''

    centralised = True

    def __init__(self,
                input_size,
                action_size,
                hidden_size = 32,
                value_function = False):
        super(CentralCritic, self).__init__()

        input_size = input_size[0]
        self.input_size = input_size
        self.value_function = value_function

        self._in_mlp = nn.Sequential(
            nn.Linear(input_size+action_size * (not value_function), hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward (self, states, actions):
        '''
        Only batch operations
        '''
        if not self.value_function:
            states = torch.cat((torch.flatten(states,start_dim=1),torch.flatten(actions,start_dim=1)),dim=1)
        return self._in_mlp(states)

class SharedCritic(nn.Module):
    '''
    Centralised critic but with multiple heads for each building.
    '''
    centralised = False

    def __init__(self,
                input_size,
                action_size,
                hidden_size = 32,
                value_function = False):
        super(SharedCritic, self).__init__()

        input_size = input_size[0]
        
        self.input_size = input_size
        self.value_function = value_function
    
        self._in_mlp = nn.Sequential(
            nn.Linear(input_size+action_size*(not value_function), hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )

    def forward (self, states, actions):
        '''
        Only batch operations
        '''
        if not self.value_function:
            states = torch.cat((torch.flatten(states,start_dim=1),torch.flatten(actions,start_dim=1)),dim=1)
        return self._in_mlp(states)