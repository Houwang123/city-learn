import numpy as np
from agents.actor_critic_agent.network import Policy
import torch as t
import torch.optim as optim
import logging
import os
import time
import random
from torch.distributions import Categorical

# t.autograd.set_detect_anomaly(True)
eps = np.finfo(np.float32).eps.item()


def normalize_observation(observation):
    assert len(observation) == 28
    observation_normalized = np.array(observation, dtype=np.float32)
    observation_normalized[0] /= 12
    observation_normalized[1] /= 31
    observation_normalized[2] /= 24
    observation_normalized[3:7] /= 30
    observation_normalized[7:11] /= 100
    observation_normalized[11:15] /= 1000
    observation_normalized[15:19] /= 1000
    # the rest is already around 0-1

    return observation_normalized


def calc_reward(observation):
    """requires un-normalized observation"""
    month, day, hour = observation[0], observation[1], observation[2]
    outdoor_temp, outdoor_temp_6h, outdoor_temp_12h, outdoor_temp_24h = \
        observation[3], observation[4], observation[5], observation[6]
    outdoor_humidity, outdoor_humidity_6h, outdoor_humidity_12h, outdoor_humidity_24h = \
        observation[7], observation[8], observation[9], observation[10]
    diffuse_solar_radiation, diffuse_solar_radiation_6h, diffuse_solar_radiation_12h, diffuse_solar_radiation_24h = \
        observation[11], observation[12], observation[13], observation[14]
    direct_solar_radiation, direct_solar_radiation_6h, direct_solar_radiation_12h, direct_solar_radiation_24h = \
        observation[15], observation[16], observation[17], observation[18]
    unit_carbon_intensity = observation[19]
    non_shiftable_load = observation[20]
    solar_generation = observation[21]
    electrical_storage_soc = observation[22]
    net_consumption = observation[23]
    electricity_pricing, electricity_pricing_6h, electricity_pricing_12h, electricity_pricing_24h = \
        observation[24], observation[25], observation[26], observation[27]

    if net_consumption < 0:
        reward = 0
        # reward = net_consumption
    else:
        reward = -electricity_pricing * net_consumption
        reward -= unit_carbon_intensity * net_consumption


    reward *= -1  # TODO: remove this line
    # # based on observation, reward has mean -0.6 and std 0.62
    # reward = (reward - (-1)) / 0.62
    return reward


class ReinforceAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """

    def __init__(self):
        self.action_space = {}
        self.policy = Policy()
        self.optimizer = t.optim.Adam(self.policy.parameters(), lr=1e-2)
        self.gamma = 0.99


        # with open('../../net_param/critic_1660483051.0458481.net', 'rb') as f:
        #     self.critic.load_state_dict(t.load(f))
        #
        # with open('../../net_param/actor_1660483051.0450208.net', 'rb') as f:
        #     self.actor.load_state_dict(t.load(f))

        print('Agent initialized')

    def set_action_space(self, agent_id, action_space):
        if agent_id == 0:
            print('episode init (set_action_space)')

            self.total_reward = 0
            self.rewards = []
            self.saved_log_probs = []

        self.action_space[agent_id] = action_space

    def finish_episode(self, agent_id, metrics, episodes_completed):
        if agent_id == 0:
            print(f'total reward: {self.total_reward}')

            R = 0
            policy_loss = []
            returns = []
            for r in self.rewards[::-1]:
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = t.tensor(returns)
            print(f'returns mean, std: {returns.mean():.2f}, {returns.std():.2f}')
            returns = (returns - returns.mean()) / (returns.std() + eps)
            policy_loss = t.tensor(0.)
            for log_prob, R in zip(self.saved_log_probs, returns):
                policy_loss += -log_prob * R
            self.optimizer.zero_grad()
            print(f'policy_loss: {policy_loss}')
            policy_loss.backward()
            self.optimizer.step()

            t.save({"weight": self.policy.state_dict()},
                   os.path.join('./net_param', f'policy_{episodes_completed}.net'))

    def compute_action(self, observation, agent_id):
        """
        1. Calculate reward, get observation
        2. Get action
        3. Update actor
        4. Update critic
        """
        reward = calc_reward(observation)
        self.rewards.append(reward)
        self.total_reward += reward
        observation_normalized = normalize_observation(observation)
        agent_id_tensor = t.zeros((5, ), dtype=t.float32)
        agent_id_tensor[agent_id] = 1
        observation_tensor = t.tensor(observation_normalized, dtype=t.float)
        observation_tensor = t.cat((observation_tensor, agent_id_tensor), dim=0)

        probs = self.policy(observation_tensor)
        c = Categorical(probs)
        choice = c.sample()
        action = -1 + 0.1 * choice.item()
        self.saved_log_probs.append(c.log_prob(choice))

        with open('reward_function.txt', 'a') as f:
            f.write(str(reward) + '\n')

        try:
            assert -1 <= action <= 1
        except AssertionError as e:
            print('action: ', action)
            print('observation: ', observation)
            print('observation_normalized: ', observation_normalized)
            raise e

        if agent_id == 0:
            logging.debug(('date:', (observation[0], observation[1], observation[2]),
                           'action: %.1f' % action,
                           'reward:', reward,
                           'ps:', [p.item() for p in probs])) # , 'observation:', observation))

        action = np.array([action], dtype=self.action_space[agent_id].dtype)
        assert self.action_space[agent_id].contains(action)
        return action
