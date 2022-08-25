column_name = [
    ('M',True), # 0
    ('D',True), # 1
    ('H',True), # 2
    ('T',True), # 3
    ('T_6h',True), # 4
    ('T_12h',True), # 5
    ('T_24h',True), # 6
    ('Hum',True), # 7 
    ('Hum_6h',True), # 8
    ('Hum_12h',True), # 9 
    ('Hum_24h',True), # 10
    ('dhi',True), # 11
    ('dhi_6h',True), # 12
    ('dhi_12h',True), # 13
    ('dhi_24h',True), # 14
    ('dni',True), # 15
    ('dni_6h',True), # 16
    ('dni_12h',True), # 17
    ('dni_24h',True), # 18
    ('carbon_intensity',True), # 19
    ('base_consumption',False), # 20
    ('solar_generation',False), # 21 
    ('electrical_storage_soc',False), # 22
    ('net_electricity_consumption',False), # 23
    ('price',False), # 24
    ('price_6h',False), # 25
    ('price_12h',False), # 26
    ('price_24h',False) # 27
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