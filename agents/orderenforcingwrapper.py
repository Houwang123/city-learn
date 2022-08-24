from agents.user_agent import UserAgent
from rewards.user_reward import UserReward
import numpy as np
from matplotlib import pyplot as plt
import os
class OrderEnforcingAgent:

    

    def __init__(self, agent = UserAgent()):
        self.num_buildings = 0
        self.agent = agent
        self.s, self.a, self.r, self.ns = None, None, None, None
        self.rewards = []
    
    def register_reset(self, observation, training=True):
        """Get the first observation after env.reset, return action""" 
        obs = observation["observation"]
        self.num_buildings = len(obs)
        self.agent.register_reset(observation, training)
        self.s, self.a, self.r, self.ns = None, None, None, None
        self.rewards = []
        self.trace = []
        return self.compute_action(obs)
    
    def raise_aicrowd_error(self, msg):
        raise NameError(msg)

    def compute_action(self, obs):
        """
        Inputs: 
            observation - List of observations from the env
            (number of buildings, observations)
        Returns:
            actions - List of actions in the same order as the observations
        """

        obs = np.array(obs, dtype=np.float32)
        # Ensure observation register reset has been called
        assert self.num_buildings is not None
        assert self.num_buildings == len(obs)
        
        # Reward calculation
        # r = UserReward(agent_count=len(obs),observation=obs).calculate()
        # TODO: Better custom reward module
        baseline_consumption = obs[:,20]-obs[:,21]
        net_consumption = obs[:,23]
        diff_consumption = (baseline_consumption - net_consumption) 
        r = np.squeeze(diff_consumption * (obs[:,19] + obs[:,24]))
        if not self.a is None:
            r = r + np.minimum(0,np.squeeze(obs[:,22])+np.squeeze(self.a)+0.1) # Penalize discharge on an empty battery

        # Compute actions
        actions = self.agent.compute_action(obs)
        if not self.s is None:
            self.agent.update(self.s,self.a,r,obs)

        # Visualization
        self.rewards.append(sum(r))
        self.trace.append((self.s,self.a,self.r,self.ns))

        self.s, self.a, self.r, self.ns = obs, actions, r, self.s
        return actions

    def save(self, path):
        # Agent
        self.agent.save(path)

        # Rewards
        fig, ax = plt.subplots(nrows=1,ncols=1)
        average = np.array(self.rewards).mean()
        smoothed_rewards = np.convolve(self.rewards, np.ones(24)/24)
        ax.plot(smoothed_rewards)
        ax.set_title('Rewards, smoothed')
        ax.axhline(y=average,color='r')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        fig.savefig(os.path.join(path,'rewards.png'))
        plt.close(fig)

        # Sample actions
        actions_taken = []
        for step in range(-48,-24):
            actions_taken.append(self.trace[step][1])
        plt.figure(figsize=(6,8),dpi=80)
        plt.imshow(np.array(actions_taken),cmap='jet',interpolation='nearest',vmin=-1,vmax=1)
        plt.colorbar()
        plt.savefig(os.path.join(path,'sample_actions.png'))
    
    def load(self, path):
        self.agent.load(path)