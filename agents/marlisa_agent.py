import enum
from xml.dom import NotFoundErr
import matplotlib.pyplot as plt
from pathlib import Path
from agents.marlisa.agents.marlisa import MARLISA
import numpy as np        
import time

def encode_uid(uid_list, uid):
    for i,u in enumerate(uid_list):
        if u == uid:
            return "Building_" + str(i + 1)
    raise NotFoundErr

def encode_uid_for_dict(uid_list, dict_of_uids):
    return {encode_uid(uid_list, uid):dict_of_uids[uid] for uid in dict_of_uids.keys()}

def transfer_building_info_format(o):
    uid_dist = {}
    uid_list = []
    assert len(o) >= 2 # Need at least 2 buildings to get all uids
    block = o[0]
    for uid in block['correlations_dhw'].keys():
        uid_dist[uid] = 1
    block = o[1]
    for uid in block['correlations_dhw'].keys():
        uid_dist[uid] = 1
    for block in o:
        for uid in uid_dist.keys():
            uid_dist[uid] = 1
        for uid in block['correlations_dhw'].keys():
            uid_dist[uid] = 2
        for uid in uid_dist.keys():
            if uid_dist[uid] == 1:
                uid_list.append(uid)
                break
    return {encode_uid(uid_list, uid):{'building_type': 1,
                        'climate_zone': -1,
                        'solar_power_capacity (kW)': 6.4,
                        'Annual_DHW_demand (kWh)': o[i]['annual_dhw_demand'],
                        'Annual_cooling_demand (kWh)': o[i]['annual_cooling_demand'],
                        'Annual_nonshiftable_electrical_demand (kWh)': o[i]['annual_nonshiftable_electrical_demand'],
                        'Correlations_DHW': encode_uid_for_dict(uid_list, o[i]['correlations_dhw']),
                        'Correlations_cooling_demand': encode_uid_for_dict(uid_list, o[i]['correlations_dhw']),
                        'Correlations_non_shiftable_load': encode_uid_for_dict(uid_list, o[i]['correlations_non_shiftable_load'])
                        } for i,uid in enumerate(uid_list)}

from gym.spaces import Box
def dict_to_action_space(aspace_dict):
    return Box(
                low = aspace_dict["low"],
                high = aspace_dict["high"],
                dtype = aspace_dict["dtype"],
              )

def transfer_spaces_format(o):
    return [dict_to_action_space(d) for d in o]

class MarlisaAgent:

    def __init__(self):
        pass

    def set_action_space(self, agent_id, action_space):
        pass    # Does not do anything. Does stuff in pass_in_all_observation instead

    def pass_in_all_observation(self, observation):
        self.obs_actions_spaces = observation["action_space"]
        self.obs_observations_spaces = observation["observation_space"]
        self.obs_building_info = observation["building_info"]

        self.actions_spaces = transfer_spaces_format(self.obs_actions_spaces)
        self.observations_spaces = transfer_spaces_format(self.obs_observations_spaces)
        self.building_info = transfer_building_info_format(self.obs_building_info)

        self.params_agent = {'building_ids':["Building_"+str(i) for i in [1,2,3,4,5]],
                        'buildings_states_actions':'agents/marlisa/buildings_state_action_space.json', 
                        'building_info':self.building_info,
                        'observation_spaces':self.observations_spaces, 
                        'action_spaces':self.actions_spaces, 
                        'hidden_dim':[256,256], 
                        'discount':0.99, 
                        'tau':5e-3, 
                        'lr':3e-4, 
                        'batch_size':256, 
                        'replay_buffer_capacity':1e5, 
                        'regression_buffer_capacity':3e4, 
                        'start_training':600, # Start updating actor-critic networks
                        'exploration_period':7500, # Just taking random actions
                        'start_regression':500, # Start training the regression model
                        'information_sharing':True, # If True -> set the appropriate 'reward_function_ma' in reward_function.py
                        'pca_compression':.95, 
                        'action_scaling_coef':0.5, # Actions are multiplied by this factor to prevent too aggressive actions
                        'reward_scaling':5., # Rewards are normalized and multiplied by this factor
                        'update_per_step':2, # How many times the actor-critic networks are updated every hourly time-step
                        'iterations_as':2,# Iterations of the iterative action selection (see MARLISA paper for more info)
                        'safe_exploration':True} 

        # Instantiating the control agent(s)
        self.agents = MARLISA(**self.params_agent)
        self.state = None

        
    def compute_all_actions(self, next_state, reward):
        """Get observation return actions for all agents, inner detail in marlisa.py"""
        if self.state is None:
            self.state = next_state
            self.j = 0
            self.is_evaluating = False
            self.action, self.coordination_vars = self.agents.select_action(self.state, deterministic=self.is_evaluating)
        else:
            self.action_next, self.coordination_vars_next = self.agents.select_action(next_state, deterministic=self.is_evaluating)
            self.agents.add_to_buffer(self.state, self.action, reward, next_state, False, self.coordination_vars, self.coordination_vars_next)
            self.coordination_vars = self.coordination_vars_next
            self.state = next_state
            self.action = self.action_next
            
            self.is_evaluating = (self.j > 3*8760)
            self.j += 1

        return self.action