import numpy as np

# Please make sure all new reward function names end with "reward", like "default_reward", and nothing else ends with "reward"

def default_reward(electricity_consumption, 
                   carbon_emission, 
                   electricity_price, 
                   agent_ids):
        
    carbon_emission = np.array(carbon_emission).clip(min=0)
    electricity_price = np.array(electricity_price).clip(min=0)
    reward = (carbon_emission + electricity_price)*-1
    
    return reward