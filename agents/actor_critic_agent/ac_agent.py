import numpy as np
from agents.actor_critic_agent.network import Actor, Critic
import torch as t
import torch.optim as optim
import logging
import os
import time

t.autograd.set_detect_anomaly(True)

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
        reward = net_consumption
    else:
        reward = -electricity_pricing * net_consumption
        reward -= unit_carbon_intensity * net_consumption

    # based on observation, reward has mean -0.6 and std 0.62
    reward = (reward - (-0.6)) / 0.62
    # print(reward)
    # print('reward: ', reward, 'net consumption: ', net_consumption, 'electricity_pricing: ', electricity_pricing)
    return reward


class ActorCriticAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """

    def __init__(self):
        self.action_space = {}
        self.actor = Actor()
        self.critic = Critic()
        self.actor_optimizer = t.optim.SGD(self.actor.parameters(), lr=1e-2)
        self.critic_optimizer = t.optim.SGD(self.critic.parameters(), lr=1e-2, momentum=0.9)
        # self.alpha_actor = 1e-2  # learning rate for actor
        # self.alpha_critic = 1e-3  # learning rate for critic
        self.gamma = 0  # discount factor used in TD-error calculation
        # self.last_pred_reward = None
        # self.last_choice_prob = None
        self.last_action = None
        self.last_observation_tensor = None
        self.last_state = None
        self.last_choice = None

        # with open('../../net_param/critic_1660483051.0458481.net', 'rb') as f:
        #     self.critic.load_state_dict(t.load(f))
        #
        # with open('../../net_param/actor_1660483051.0450208.net', 'rb') as f:
        #     self.actor.load_state_dict(t.load(f))

        print('Agent initialized')


    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

        # save actor and critic parameters
        if agent_id == 0:
            t.save({"weight": self.actor.state_dict()}, os.path.join('./net_param', f'actor_{time.time()}.net'))
            t.save({"weight": self.critic.state_dict()}, os.path.join('./net_param', f'critic_{time.time()}.net'))

    def evaluate_Q(self, observation, action):
        """calculate Q(s, a)"""
        critic_input = t.cat((t.tensor(observation), action.unsqueeze(0)))
        return self.critic(critic_input)

    def compute_action(self, observation, agent_id):
        """
        1. Calculate reward, get observation
        2. Get action
        3. Update actor
        4. Update critic
        """
        reward = calc_reward(observation)
        observation_normalized = normalize_observation(observation)
        observation_tensor = t.tensor(observation_normalized, dtype=t.float)

        ps = t.softmax(self.actor(observation_tensor), dim=0)
        choice = t.argmax(ps, dim=0)
        # choice_prob = ps[choice]
        action = -1 + 0.1 * choice
        action = action.detach()

        q = self.evaluate_Q(observation_normalized, action)

        with open('reward_function.txt', 'a') as f:
            f.write(str(reward) + '\n')

        try:
            assert -1 <= action <= 1
        except AssertionError as e:
            print('action: ', action)
            print('observation: ', observation)
            print('observation_normalized: ', observation_normalized)
            raise e


        if self.last_state is not None:
            last_ps = t.softmax(self.actor(self.last_observation_tensor), dim=0)
            last_choice_prob = last_ps[self.last_choice]
            actor_loss = t.log(last_choice_prob)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # for actor_param in self.actor.parameters():
            #     # print(self.alpha_actor, reward, actor_param.grad)
            #     actor_param.data += self.alpha_actor * reward * actor_param.grad
            #     actor_param.grad = None

            # print(self.last_pred_reward.requires_grad)
            last_pred_reward = self.evaluate_Q(self.last_state, self.last_action)
            critic_loss = (reward + self.gamma * q.detach() - last_pred_reward).pow(2)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # print('action:', action.item(), 'q:', q[0].item(),
            #       'reward:', reward, 'critic loss:', critic_loss.item(), 'ps:', ps.detach().numpy(), 'observation:', observation)
            if agent_id == 0:
                logging.debug(('action:', action.item(), 'q:', q[0].item(),
                      'reward:', reward, 'critic loss:', critic_loss.item(), 'last pred reward', last_pred_reward.item()
                               , 'ps:', [p.item() for p in ps], 'observation:', observation))

        # self.last_pred_reward = q
        # self.last_choice_prob = choice_prob
        self.last_action = action
        self.last_choice = choice
        self.last_observation_tensor = observation_tensor
        self.last_state = observation_normalized

        action = action.item()
        action = np.array([action], dtype=self.action_space[agent_id].dtype)
        assert self.action_space[agent_id].contains(action)
        return action
