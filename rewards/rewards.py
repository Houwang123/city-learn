import numpy as np

# Please make sure all new reward function names end with "reward", like "default_reward", and nothing else ends with "reward"

#########################################################
# OFFICIAL USER_REWARD STRUCTURE                        #
#########################################################

'''
def default_reward(electricity_consumption, 
                   carbon_emission, 
                   electricity_price, 
                   agent_ids):
        
    carbon_emission = np.array(carbon_emission).clip(min=0)
    electricity_price = np.array(electricity_price).clip(min=0)
    reward = (carbon_emission + electricity_price)*-1
    
    return reward
'''

#########################################################
# BELOW ARE REWARDS THAT USE ALL AVAILABLE OBSERVATIONS #
# NOT OFFICIALLY ALLOWED - I DON'T SEE WHY NOT          #
#########################################################


def simple_reward(state, next_state, actions):
    '''
    Tries to correlate to metric while penalizing illegal actions
    '''

    # Main metric
    state, next_state, actions = map(np.array, [state,next_state,actions])
    base_consumption = np.maximum(0,next_state[:,20] - next_state[:,21])
    net_consumption = np.maximum(0,next_state[:,23])
    
    reward = (base_consumption-net_consumption) * (1.8 * next_state[:,19] + next_state[:,24])
    reward = reward.squeeze()
    # Illegal actions
    previous_battery_soc = state[:,22].squeeze()
    actions = actions.squeeze()
        # Discharging too much
    reward = reward + 0.5*np.minimum(0,previous_battery_soc+(actions+0.2))
        # Charging too much
    reward = reward + 0.5*np.minimum(0, (1 - previous_battery_soc)-(actions-0.2))
    
    # Output
    
    return reward

def cubic_simple_reward(state, next_state, actions):
    return simple_reward(state, next_state, actions) ** 3