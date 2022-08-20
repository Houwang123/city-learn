import torch
import torch.nn as nn
import torch.nn.functional as F
import random, math
import typing
import time

class CommNet(nn.Module):
    '''
    Implements CommNet for a single building
    Of the CityLearn challenge
    LSTM version with skip connection for the final layer

    'https://proceedings.neurips.cc/paper/2016/file/55b1927fdafef39c48e5b73b5d61ea60-Paper.pdf'
    '''
    
    def __init__(
                self, 
                input_size,         # Observation accessible to each building (assuming homogenous)
                hidden_size = 32,   # Size of each hidden layer
                memory_size = 10,   # Hidden vector accessible at each communication step
                comm_size = 4,      # Number of communication channels
                comm_steps = 2,     # Number of communication steps
                device = 'cpu',      # GPU or CPU (default CPU)  
                ):
                
        super(CommNet, self).__init__()

        self.device = device
        self.agent_number = input_size[0]
        input_size = input_size[1]
        self.input_size = input_size
        self.comm_size = comm_size
        self.comm_steps = comm_steps
        self.is_first_time = True

        # Calculate first hidden layer 
        self._in_mlp = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size,memory_size)
        )

        # Communication 
        self._lstm = nn.LSTMCell(
            input_size = comm_size,
            hidden_size = memory_size
        )

        self._comm_mlp = nn.Sequential(
            nn.Linear(memory_size,hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size,comm_size)
        )

        # Output
        # Calculate based on inputs and final memory
        self._out_mlp = nn.Sequential(
            nn.Linear(input_size+memory_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )


    def forward(self,x : torch.Tensor, batch = False):

        out = None
        if not batch:

            # (Building, Observations)
            
            # Initial hidden states
            start = time.time()
            hidden_states = self._in_mlp(x)
            cell_states = torch.zeros(hidden_states.shape,device=self.device)
            # Communication
            start = time.time()
            for t in range(self.comm_steps):
                # Calculate communication vectors
                comm = self._comm_mlp(hidden_states)
                total_comm = torch.sum(comm,0)
                comm = (total_comm - comm) / (self.agent_number-1)
                # Apply LSTM   
                hidden_states, cell_states = self._lstm(comm,(hidden_states,cell_states))
            
            out = self._out_mlp(torch.cat((x,hidden_states),dim=1))
        else:
            # (Batch, Building, Observation)
            assert(len(x.size()) == 3) # Expecting dimension (Batch, Building, Observation)
            n_batch = x.size(0)
            n_building = x.size(1)
            n_observation = x.size(2)

            x = torch.reshape(x, [n_batch * n_building, n_observation]) # Turn it into (Batch * Building, Observation)

            hidden_states = self._in_mlp(x)
            cell_states = torch.zeros(hidden_states.shape,device=self.device)

            # Communication
            for t in range(self.comm_steps):
                # Calculate communication vectors
                comm = self._comm_mlp(hidden_states)
                total_comm = torch.sum(comm,0)
                comm = (total_comm - comm) / (self.agent_number-1)
                # Apply LSTM   
                hidden_states, cell_states = self._lstm(comm,(hidden_states,cell_states))
            
            out = self._out_mlp(torch.cat((x,hidden_states),dim=1))
            
            out = torch.reshape(out, [n_batch, n_building, 1]) # Turn "out" from (Batch * Building, 1) back into (Batch, Building, 1)

            # Reshape time save: 0.0147 -> 0.0010 (on average, second, per operation)

            

        return out

    def to(self,device):
        super().to(device)
        self.device = device