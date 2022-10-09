import numpy as np

def daytime_phase(month, day, hour):
    day_length = {
        i:[10,15] for i in range(1,13)
    }
    assert month>=1 and month<=12
    [day_hour_left_bound,day_hour_right_bound] = day_length[month]
    if day_hour_left_bound<=hour and hour<=day_hour_right_bound:
        return 1
    else:
        return 0

class RBCAgent():
    def separate_rbc_policy(self, observation, agent_id):
        month = observation[0]
        day = observation[1]
        hour = observation[2]
        daytime_phase_value = daytime_phase(month, day, hour)
        
        action = 0.0 # Default value
        if daytime_phase_value == 1:
            if agent_id == 4:
                action = -0.04 + 0.00193223620844446
            else:
                action = 0.24 + -0.005077212243585316
        elif daytime_phase_value == 0:
            action = -0.1 + -0.015223804448657952
        return action

    def rbc_policy(self, observation):
        """
        Simple rule based policy based on day or night time
        """
        actions = np.array([[self.separate_rbc_policy(observation[id], id)] for id in range(5)], dtype=np.float64)
        return actions

    def __init__(self):
        pass

    def register_reset(self, observation, training = True):
        pass

    def compute_action(self, observation):
        """Get observation return action"""
        return self.rbc_policy(observation)

    def update(self, s, a, r, ns): 
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass