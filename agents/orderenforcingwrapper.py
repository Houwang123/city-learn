from agents.user_agent import UserAgent
from rewards.user_reward import UserReward
import numpy as np
class OrderEnforcingAgent:

    s, a, r, ns = None, None, None, None

    def __init__(self, agent = UserAgent()):
        self.num_buildings = 0
        self.agent = agent
    
    def register_reset(self, observation, training=True):
        """Get the first observation after env.reset, return action""" 
        obs = observation["observation"]
        self.num_buildings = len(obs)
        self.agent.register_reset(observation, training)
        self.s, self.a, self.r, self.ns = None, None, None, None
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

        
        rewards = UserReward(agent_count=len(obs),observation=obs).calculate()
        actions = self.agent.compute_action(obs)
        if not self.s is None:
            self.agent.update(self.s,self.a,rewards,obs)
        self.s, self.a = obs, actions
        return actions
