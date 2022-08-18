import numpy as np

class NothingAgent:
    '''
    Does nothing
    '''

    def register_reset(*args):
        pass

    def update(*args):
        pass

    def compute_action(obs):
        return np.array([0 for x in range(len(obs))])
