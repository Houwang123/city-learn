from abc import ABC, abstractmethod
import numpy as np
import math
# Example usage 
# engineer = DecoratorA(DecoratorB(BaseFeatureEngineer(observations)))
# shape = engineer.out
# result = engineer.tranform(obs)

class FeatureEngineer(ABC):

    def __init__(self,input_feature_engineer):
        self.input_feature_engineer = input_feature_engineer

    @property
    @abstractmethod
    def out(self):
        # Should be either be constant or dependent on out of input
        pass

    @abstractmethod 
    def _transform(self, *args):
        pass

    def transform(self, x, *args):
        '''
        Transforms a single observation 
        '''
        return self._transform(self.input_feature_engineer.transform(x), *args)

    def batch_transform(self, x, *args):
        '''
        Transform but with an extra first dimension.
        Can overwrite to keep vectorization performance boost.
        '''
        for i in range(len(x)):
            x[i] = self.transform(x[i])

    def register_reset(self, observation):
        self.observation = observation
        self.input_feature_engineer.register_reset(observation)

class BaseFeatureEngineer():

    observation = None
    def register_reset(self, observation):
        self._out = np.array(observation['observation']).shape
        self.observation = observation

    @property
    def out(self):
        return self._out
        
    def transform(self,x, *args):
        return np.array(x,dtype=np.float32)
    

#############################################################################

class CentralCriticEngineer(FeatureEngineer):
    '''
    Flatterns observations for a central critic and joins 
    state with action
    '''

    @property
    def out(self):
        shape = self.input_feature_engineer.out
        return (math.prod(shape),)
    
    def _transform(self, x):
        return x.flatten()