from citylearn.citylearn import CityLearnEnv

def str_to_class(classname):
    #return getattr(sys.modules[__name__], classname)
    return eval(classname)

def deserialize_document(doc):
    if isinstance(doc, dict):
        for k, v in doc.items():
            if isinstance(v, str):
                if not(v[0] == "'" and v[-1] == "'"):
                    doc[k] = str_to_class(v)
                else:
                    doc[k] = v[1:-1]
            elif isinstance(v, list) or isinstance(v, dict):
                deserialize_document(v)
    elif isinstance(doc,list):
        for i,v in enumerate(doc):
            for k, v in doc:
                if isinstance(v, str):
                    if not(v[0] == "'" and v[-1] == "'"):
                        doc[i] = str_to_class(v)
                    else:
                        doc[k] = v[1:-1]
                elif isinstance(v, list) or isinstance(v, dict):
                    deserialize_document(v)



##################################################
# Environment                                    #
##################################################

def action_space_to_dict(aspace):
    """ Only for box space """
    return { "high": aspace.high,
             "low": aspace.low,
             "shape": aspace.shape,
             "dtype": str(aspace.dtype)
    }

def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    building_info = env.get_building_information()
    building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                "building_info": building_info,
                "observation": observations }
    return obs_dict

def setup_environmnent(schema_path):
    return CityLearnEnv(schema_path)

##################################################
# Rewards                                        #
##################################################

from rewards.rewards import *
from rewards import get_reward

def setup_reward(reward):
    get_reward.reward_function = reward

##################################################
# Agents                                         #
##################################################
from agents.orderenforcingwrapper import OrderEnforcingAgent

from agents.agents.ddpg import DDPGAgent, TD3Agent
from agents.agents.ppo import PPOAgent
from agents.agents.rules import NothingAgent
from agents.networks.central_critic import *
from agents.networks.comm_net import *
from agents.features import *

def setup_agent(type, attributes):
    return OrderEnforcingAgent(agent=type(**attributes))
