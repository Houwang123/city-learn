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
        self.r = np.zeros(add_dimension(memory_size,(act_shape[0],)),dtype=np.float32)
        self.c_ns = np.zeros(add_dimension(memory_size,critic_obs_shape),dtype=np.float32)
        self.a_ns = np.zeros(add_dimension(memory_size,actor_obs_shape),dtype=np.float32)

        self.size, self.i, self.MAX_SIZE = 0,0, memory_size
        self.device = 'cpu'

    def add(self, s, a, r, ns):
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

        self.num_agents = 0
        self.has_setup = False

    def register_reset(self, observation, training=True):
        '''
        Called at the start of each new episode
        '''
        if not self.has_setup:
            self.step = 0
            self.has_setup = True
            self.training = training
            self.num_agents = len(observation["observation"])
            
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
            # Someone needs to be assigned to this
            action = action + np.random.normal(scale=min(0.8,max(0.05,5*np.exp(-0.0002*self.step))), size=action.shape)
            action = np.clip(action,a_min=-1.0,a_max=1.0)
        return action

    def update(self,s,a,r,ns):
        self.rb.add(s,a,r,ns)
        self.step += 1
        
        if len(self.rb) < self.BATCH_SIZE: # Get some data first before beginning the training process
            return

        a_s, c_s,a,r,a_ns, c_ns = self.rb.sample(batch_size=self.BATCH_SIZE)

        if self.critic_setup.centralised:
            r = torch.unsqueeze(torch.sum(r,1),1)

        # Critic update
        nsa, y_t = None,None
        with torch.no_grad():
            nsa = self.actor_target.forward(a_ns)
            y_t = torch.add(r, self.GAMMA * self.critic_target(c_ns,nsa))
        self.c_optimize.zero_grad()
        y_c = self.critic(c_s, a) 
        c_loss = self.c_criterion(y_c,y_t)
        c_loss.backward()
        self.c_optimize.step()   

        # Actor update
        self.a_optimize.zero_grad()
        a_loss = -torch.sum(self.critic(c_s,self.actor.forward(a_s)),dim=1).mean() # Maximize gradient direction increasing objective function
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


class TD3Agent(DDPGAgent):

    def __init__(self, actor, critic, actor_feature, critic_feature, a_kwargs={}, c_kwargs={}, gamma=0.99, lr=0.0003, tau=0.001, batch_size=64, memory_size=4096, device='cpu', clip_size = 0.05):
        super().__init__(actor, critic, actor_feature, critic_feature, a_kwargs, c_kwargs, gamma, lr, tau, batch_size, memory_size, device)
        self.clip_size = clip_size

    def register_reset(self, observation, training = True):

        if not self.has_setup:
            self.step = 0
            self.has_setup = True
            self.training = training
            self.num_agents = len(observation["observation"])
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

            self.critic_2 = self.critic_setup(input_size = self.critic_feature.out, action_size=self.num_agents, **self.c_kwargs)
            self.critic_target_2 = copy.deepcopy(self.critic_2)
            self.c_optimize_2 = optim.Adam(self.critic_2.parameters(),lr=self.LR)
            
            self.to(self.device)

    def to(self,device):
        super(TD3Agent,self).to(device)
        self.critic_2.to(device)
        self.critic_target_2.to(device)

    def update(self, s, a, r, ns):
        self.rb.add(s,a,r,ns)
        self.step += 1

        if len(self.rb) < self.BATCH_SIZE: # Get some data first before beginning the training process
            return

       
        a_s, c_s, a, r,a_ns, c_ns = self.rb.sample(batch_size=self.BATCH_SIZE)

        if self.critic_setup.centralised:
            r = torch.unsqueeze(torch.sum(r,1),1)

            
        # Critic update
        nsa, y_t = None,None
        with torch.no_grad():
            nsa = self.actor_target.forward(a_ns)

            # Target policy smoothing
            nsa = nsa + torch.normal(mean = torch.zeros(nsa.size()),
                                     std = torch.ones(nsa.size())*self.clip_size)
            nsa = torch.clip(nsa,-1,1)

            # Clipped double-Q learning
            y_t = torch.add(r, 
                            self.GAMMA * torch.minimum(
                                self.critic_target(c_ns,nsa),
                                self.critic_target_2(c_ns,nsa)))

        self.c_optimize.zero_grad()
        self.c_optimize_2.zero_grad()
        y_c = self.critic(c_s, a)
        y_c_2 = self.critic_2(c_s, a) 
        c_loss = self.c_criterion(y_c,y_t)
        c_loss.backward()
        self.c_optimize.step()   
        c_loss_2 = self.c_criterion(y_c_2,y_t)
        c_loss_2.backward()
        self.c_optimize_2.step()
 
        
        # Actor and target network update  
        if self.step % 2 == 0: # Delayed policy update

            self.a_optimize.zero_grad()
            a_loss = -torch.sum(self.critic(c_s,self.actor.forward(a_s)),dim=1).mean() # Maximize gradient direction increasing objective function

            a_loss.backward()
            self.a_optimize.step()

            for ct_p, c_p in zip(self.critic_target.parameters(), self.critic.parameters()):
                ct_p.data = ct_p.data * (1.0-self.TAU) + c_p.data * self.TAU
            for ct_p, c_p in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
                ct_p.data = ct_p.data * (1.0-self.TAU) + c_p.data * self.TAU
            for at_p, a_p in zip(self.actor_target.parameters(), self.actor.parameters()):
                at_p.data = at_p.data * (1.0-self.TAU) + a_p.data * self.TAU

    def save(self, path):
        torch.save(self.actor.state_dict(),os.path.join(path,'actor.pt'))
        torch.save(self.critic.state_dict(),os.path.join(path,'critic_1.pt'))
        torch.save(self.critic_2.state_dict(),os.path.join(path,'critic_2.pt'))

    def load(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path,'actor.pt')))
        self.critic.load_state_dict(torch.load(os.path.join(path,'critic_1.pt')))
        self.critic_2.load_state_dict(torch.load(os.path.join(path,'critic_2.pt')))