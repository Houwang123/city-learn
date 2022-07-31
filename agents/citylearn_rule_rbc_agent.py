import numpy as np


def rbc_policy(observation, action_space):
    """
    Simple rule based policy based on day or night time
    """
    hour = observation[2]  # Hour index is 2 for all observations

    action = 0.0  # Default value
    if 7 <= hour <= 15:
        action = -0.05
    elif 16 <= hour <= 18:
        action = -0.11
    elif 19 <= hour <= 22:
        action = -0.06
    elif 23 <= hour <= 24:
        action = 0.085
    elif 1 <= hour <= 6:
        action = 0.1383

    action = np.array([action], dtype=action_space.dtype)
    assert action_space.contains(action)
    return action


class MyRBCAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """

    def __init__(self):
        self.action_space = {}

    def register_reset(self, observation, action_space, agent_id):
        """Get the first observation after env.reset, return action"""
        self.action_space[agent_id] = action_space
        return rbc_policy(observation, action_space)

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        if agent_id == 0:
            print(observation)
        return rbc_policy(observation, self.action_space[agent_id])