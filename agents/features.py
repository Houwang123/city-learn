column_name = [
    ('M',True),
    ('D',True),
    ('H',True),
    ('T',True),
    ('T_6h',True),
    ('T_12h',True),
    ('T_24h',True),
    ('Hum',True), 
    ('Hum_6h',True), 
    ('Hum_12h',True),    
    ('Hum_24h',True), 
    ('dhi',True), 
    ('dhi_6h',True), 
    ('dhi_12h',True), 
    ('dhi_24h',True),
    ('dni',True), 
    ('dni_6h',True),
    ('dni_12h',True), 
    ('dni_24h',True),
    ('carbon',True), #index 19
    ('consumption',False), # index 20
    ('generation',False),  
    ('storage',False),
    ('nconsumption',False),
    ('price',False), 
    ('price_6h',False),
    ('price_12h',False), 
    ('price_24h',False)
]

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

from sklearn.preprocessing import MinMaxScaler
class MinMaxNormalizationEngineer(FeatureEngineer):
    '''
    Min-max normalization
    '''
    def __init__(self, input_feature_engineer):
        super(MinMaxNormalizationEngineer, self).__init__(input_feature_engineer)
        assert(isinstance(input_feature_engineer,BaseFeatureEngineer))
        

    def register_reset(self, observation):
        super(MinMaxNormalizationEngineer, self).register_reset(observation)
        self.scalar = MinMaxScaler()
        observation_space = observation['observation_space'][0]
        low, high = observation_space['low'],observation_space['high']
        self.scalar.fit([low,high])

    @property
    def out(self):
        return self.input_feature_engineer.out
    
    def _transform(self, x):
        return self.scalar.transform(x)

class CentralCriticEngineer(FeatureEngineer):
    '''
    Flatterns observations for a central critic and joins 
    state with action
    '''

    @property
    def out(self):
        assert self.input_feature_engineer.out == (5,28)
        return (20+8*5,)
    
    def _transform(self, x):
        return np.concatenate((x[0,:20],x[:,20:].flatten()),axis=None)