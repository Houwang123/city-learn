import numpy as np


#########################################################
# OFFICIAL USER_REWARD STRUCTURE                        #
#########################################################

def default_reward(electricity_consumption, 
                   carbon_emission, 
                   electricity_price, 
                   agent_ids):
        
    carbon_emission = np.array(carbon_emission).clip(min=0)
    electricity_price = np.array(electricity_price).clip(min=0)
    reward = (carbon_emission + electricity_price)*-1
    
    return reward

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
    net_consumption = np.maximum(0,next_state[:,23])

    reward = (-net_consumption) * (next_state[:,21] + next_state[:,24])
    reward = reward.squeeze()

    # Illegal actions
    previous_battery_soc = state[:,22].squeeze()
    actions = actions.squeeze()
        # Discharging too much
    reward = reward + np.minimum(0,previous_battery_soc+(actions+0.1))
        # Charging too much
    reward = reward + np.minimum(0, (1 - previous_battery_soc)-(actions-0.1))
    


    # Output
    
    return reward