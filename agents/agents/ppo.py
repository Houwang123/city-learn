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
    Minibatch replay buffer for PPO. Clears every time after access to ensure on policy updates.
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

    @property
    def full(self):
        if len(self) == self.MAX_SIZE-1:
            return True
        return False

    def sample(self):
        sample = [
            self.a_s[:self.size],
            self.c_s[:self.size],
            self.a[:self.size],
            self.r[:self.size],
            self.a_ns[:self.size],
            self.c_ns[:self.size]
        ]
        sample = [torch.from_numpy(x).to(self.device) for x in sample]

        self.i = 0
        self.a_s = np.zeros(self.a_s.shape,dtype=np.float32)
        self.c_s = np.zeros(self.c_s.shape,dtype=np.float32)
        self.a = np.zeros(self.a.shape,dtype=np.float32)
        self.r = np.zeros(self.r.shape,dtype=np.float32)
        self.a_ns = np.zeros(self.a_ns.shape,dtype=np.float32)
        self.c_ns = np.zeros(self.c_ns.shape,dtype=np.float32)

        self.size, self.i = 0,0

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

class PPOAgent():

    def __init__(self,
                 actor,
                 critic,
                 actor_feature, 
                 critic_feature,
                 a_kwargs = {},
                 c_kwargs = {},
                 gamma = 0.99,
                 lr = 3e-4,
                 n_epochs=10,
                 gae = 0.95,
                 minibatch_size = 1024,
                 clip=0.2,
                 device = 'cpu'):

        self.GAMMA = gamma
        self.LR = lr
        self.GENERALIZED_ADVANTAGE_ESTIMATE = gae
        self.minibatch_size = minibatch_size
        self.n_epochs = n_epochs
        self.clip = clip
        self.device = device
        self.actor_feature = actor_feature
        self.critic_feature = critic_feature
        self.a_kwargs = a_kwargs
        self.c_kwargs = c_kwargs
        self.actor_setup = actor
        self.critic_setup = critic

        self.num_agents = -1

    def register_reset(self, observation, training = True):
        
        if self.num_agents != len(observation["observation"]):
            self.step = 0
            self.training = training
            self.num_agents = len(observation["observation"])
            
            self.actor_feature.register_reset(observation)
            self.critic_feature.register_reset(observation)

            self.actor = self.actor_setup(input_size = self.actor_feature.out, **self.a_kwargs)

            # By adjusting action size to 0, a critic function becomes equivalent to a value function
            self.critic = self.critic_setup(input_size = self.critic_feature.out, action_size=self.num_agents, value_function = True,**self.c_kwargs)
            self.rb = ReplayBuffer((self.num_agents,1),self.actor_feature,self.critic_feature,memory_size=self.minibatch_size)
            
            self.c_criterion = nn.MSELoss()
            self.a_optimize = optim.Adam(self.actor.parameters(),lr=self.LR)
            self.c_optimize = optim.Adam(self.critic.parameters(),lr=self.LR)

            self.to(self.device)
        else:
            self.train()

    def to(self,device):
        self.device = device
        self.actor.to(device)
        self.critic.to(device)

    def compute_action(self,obs):
        a_obs = torch.from_numpy(self.actor_feature.transform(obs))
        a_obs = a_obs.to(device=self.device)
        with torch.no_grad():
            distribution = self.actor(a_obs)
        action = distribution.sample().cpu().numpy()[0]
        return action

    def update(self, s, a, r, ns): 
        self.rb.add(s,a,r,ns)
        
        if not self.rb.full:
            # Replay buffer not full
            return

        self.train()        
    
    def train(self):
        """
        Updates network with available data in memory
        
        Assumes in order

        #TODO: Without any assumption of the concept of episode *doneness* in the current framework
        Convergence guarantees fail
        But also we want to enable online training "mid episode"
        More research required. Write now addressing with value function, but will introduce bias,
        yet hopefully small with sufficient memory size and discount factor
        """
        a_s, c_s, a, r,a_ns, c_ns = self.rb.sample()

        if self.critic_setup.centralised:
            r = torch.unsqueeze(torch.sum(r,1),1)
        
        # Update value functions
        # Using a weird Monte Carlo learning process, but using TD for final step
        # As not assuming anything related to episode end
        y_t = torch.zeros(r.shape)
        y_t[-1] = r[-1] + self.GAMMA * self.critic(c_ns[-1],None)
        for i in range(-2,-len(r)-1,-1):
            y_t[i] = self.GAMMA * y_t[i+1] + r[i]
        y_t = y_t.detach()

        for epoch in range(self.n_epochs):
            self.c_optimize.zero_grad()
            y_c = self.critic(c_s,None)
            c_loss = self.c_criterion(y_c,y_t)
            c_loss.backward()
            self.c_optimize.step()

        # Calculate advantage
        trace = torch.zeros(r.shape)
        trace = r + self.GAMMA * self.critic(c_ns,None).detach() - self.critic(c_s, None).detach()

        a_t = torch.zeros(r.shape)
        gae = self.GENERALIZED_ADVANTAGE_ESTIMATE * self.GAMMA
        
        a_t[-1] = trace[-1]
        for i in range(-2,-len(r)-1,-1):
            a_t[i] = gae * a_t[i+1] + trace[i]
        
        # Update policy with PPO-Clip
        distributions = self.actor(a_s)
        log_probs = distributions.log_prob(a).detach()

        
        # print(a_t.mean(),trace.mean())
        for epoch in range(self.n_epochs):
            self.a_optimize.zero_grad()
            distributions = self.actor(a_s)
            curr_log_probs = distributions.log_prob(a)
            ratios = curr_log_probs-log_probs
            if self.critic_setup.centralised:
                ratios = torch.sum(ratios,1)
            ratios = torch.exp(ratios).squeeze()
            a_loss = -torch.minimum(ratios*a_t, (torch.clamp(ratios, 1-self.clip, 1+self.clip) * a_t)).mean()
            a_loss.backward()
            self.a_optimize.step()


    def save(self, path):
        torch.save(self.actor.state_dict(),os.path.join(path,'actor.pt'))
        torch.save(self.critic.state_dict(),os.path.join(path,'critic_1.pt'))

    def load(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path,'actor.pt')))
        self.critic.load_state_dict(torch.load(os.path.join(path,'critic_1.pt')))