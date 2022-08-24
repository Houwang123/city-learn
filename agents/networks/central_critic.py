import torch
import torch.nn as nn
import torch.nn.functional as F
import random, math
import typing
import numpy as np

class CentralCritic(nn.Module):
    '''
    Implements a centralised critic 
    with a multi layer perceptron
    '''

    def __init__(self,
                input_size,
                action_size,
                hidden_size = 32):
        super(CentralCritic, self).__init__()

        input_size = input_size[0]
        self.input_size = input_size


        self._in_mlp = nn.Sequential(
            nn.Linear(input_size+action_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward (self, states, actions):
        '''
        Only batch operations
        '''
        states = torch.cat((torch.flatten(states,start_dim=1),torch.flatten(actions,start_dim=1)),dim=1)
        return self._in_mlp(states)