import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time, os


def add_dimension(length, shape = None):
    return (length, *shape)

class ReplayBuffer:
    '''
    FIFO Replay Buffer for DDPG
    Because of the continuous nature of the environment, ignore done for simplicity.
    '''
    
    def __init__(self, act_shape, actor_feature, critic_feature, memory_size=1024):
        critic_obs_shape = critic_feature.out
        actor_obs_shape = actor_feature.out
        self.actor_feature = actor_feature
        self.critic_feature = critic_feature

        self.c_s = np.zeros(add_dimension(memory_size,critic_obs_shape),dtype=np.float32)
        self.a_s = np.zeros(add_dimension(memory_size,actor_obs_shape),dtype=np.float32)
        self.a = np.zeros(add_dimension(memory_size,act_shape),dtype=np.float32)
        self.r = np.zeros(memory_size,dtype=np.float32)
        self.c_ns = np.zeros(add_dimension(memory_size,critic_obs_shape),dtype=np.float32)
        self.a_ns = np.zeros(add_dimension(memory_size,actor_obs_shape),dtype=np.float32)

        self.size, self.i, self.MAX_SIZE = 0,0, memory_size
        self.device = 'cpu'

    def add(self, s, a, r, ns):
        r = np.sum(r)
        self.c_s[self.i] = self.critic_feature.transform(s)
        self.a_s[self.i] = self.actor_feature.transform(s)
        self.a[self.i] = a
        self.r[self.i] = r
        self.c_ns[self.i] = self.critic_feature.transform(ns)
        self.a_ns[self.i] = self.actor_feature.transform(ns)

        self.i += 1
        if self.i >= self.MAX_SIZE:
            self.i = 0
        elif self.i > self.size:
            self.size = self.i    

    def sample(self, batch_size = 32, t = True):
        idxs = np.random.randint(
            low = 0,
            high = self.size,
            size = batch_size
        )
        sample = [
            self.a_s[idxs],
            self.c_s[idxs],
            self.a[idxs],
            self.r[idxs],
            self.a_ns[idxs],
            self.c_ns[idxs]
        ]
        if t:
            sample = [torch.from_numpy(x).to(self.device) for x in sample]
        return sample

    def __len__(self):
        return self.size

    def __getitem__(self,i):
        '''
        Returns tuple (state, action, reward, next state)(
        '''
        if i >= self.size:
            raise IndexError
        return (self.s[i], self.a[i],self.r[i],self.ns[i])

    def to(self,device):
        self.device = device


class DDPGAgent:
    '''
    Implements base DDPG with Gaussian exploration
    '''
    
    def __init__(self,
                 actor,
                 critic,
                 actor_feature,
                 critic_feature,
                 a_kwargs = {},
                 c_kwargs = {},
                 gamma = 0.99, 
                 lr = 3e-4,
                 tau = 0.001,
                 batch_size = 64,
                 memory_size = 4096,
                 device = 'cpu'):
        self.GAMMA=gamma
        self.LR=lr
        self.BATCH_SIZE=batch_size
        self.MEMORY_SIZE=memory_size
        self.TAU=tau
        self.device = device
        self.actor_feature = actor_feature
        self.critic_feature = critic_feature
        self.actor_setup = actor
        self.critic_setup = critic
        self.a_kwargs = a_kwargs
        self.c_kwargs = c_kwargs
        self.training = True
        self.step = 0

    def register_reset(self, observation, training=True):
        '''
        Called at the start of each new episode
        '''
        self.training = training
        self.num_agents = len(observation["observation"])
        self.step = 0

        self.actor_feature.register_reset(observation)
        self.critic_feature.register_reset(observation)
        self.rb = ReplayBuffer((self.num_agents,1),self.actor_feature,self.critic_feature,memory_size=self.MEMORY_SIZE)

        self.actor = self.actor_setup(input_size = self.actor_feature.out, **self.a_kwargs)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = self.critic_setup(input_size = self.critic_feature.out, action_size=self.num_agents, **self.c_kwargs)
        self.critic_target = copy.deepcopy(self.critic)
        
        self.c_criterion = nn.MSELoss()
        self.a_optimize = optim.Adam(self.actor.parameters(),lr=self.LR)
        self.c_optimize = optim.Adam(self.critic.parameters(),lr=self.LR)

        self.to(self.device)

    def to(self,device):
        self.device = device
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
        self.rb.to(device)

    def compute_action(self, obs):
        a_obs = torch.from_numpy(self.actor_feature.transform(obs))
        a_obs = a_obs.to(device=self.device)
        with torch.no_grad():
            action = self.actor(a_obs).cpu().numpy()[0]
        if self.training:
            action = action + np.random.normal(scale=max(0.03,0,5*np.exp(-0.001*self.step)), size=action.shape)
            action = np.clip(action,a_min=-1.0,a_max=1.0)
        return action

    def update(self,s,a,r,ns):
        self.rb.add(s,a,r,ns)
        
        if len(self.rb) < self.BATCH_SIZE: # Get some data first before beginning the training process
            return

        a_s, c_s,a,r,a_ns, c_ns = self.rb.sample(batch_size=self.BATCH_SIZE)

        # Critic update
        self.c_optimize.zero_grad()
        nsa, y_t = None,None
        with torch.no_grad():
            nsa = self.actor_target.forward(a_ns)
            y_t = torch.add(torch.unsqueeze(r,1), self.GAMMA * self.critic_target(c_ns,nsa))
        y_c = self.critic(c_s, a) 
        c_loss = self.c_criterion(y_c,y_t)
        c_loss.backward()
        self.c_optimize.step()   

        # Actor update
        self.a_optimize.zero_grad()
        a_loss = -self.critic(c_s,self.actor.forward(a_s)).mean() # Maximize gradient direction increasing objective function
        a_loss.backward()
        self.a_optimize.step()

        # Target networks
        for ct_p, c_p in zip(self.critic_target.parameters(), self.critic.parameters()):
            ct_p.data = ct_p.data * (1.0-self.TAU) + c_p.data * self.TAU
        for at_p, a_p in zip(self.actor_target.parameters(), self.actor.parameters()):
            at_p.data = at_p.data * (1.0-self.TAU) + a_p.data * self.TAU

    def save(self, path):
        torch.save(self.actor.state_dict(),os.path.join(path,'actor.pt'))
        torch.save(self.critic.state_dict(),os.path.join(path,'critic.pt'))

    def load(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path,'actor.pt')))
        self.critic.load_state_dict(torch.load(os.path.join(path,'critic.pt')))