column_name = [
    ('M',True), # 0
    ('D',True), # 1
    ('H',True), # 2
    ('T - temperature',True), # 3
    ('T_6h',True), # 4
    ('T_12h',True), # 5
    ('T_24h',True), # 6
    ('Hum - humidity',True), # 7 
    ('Hum_6h',True), # 8
    ('Hum_12h',True), # 9 
    ('Hum_24h',True), # 10
    ('dhi - diffuse solar irradiance',True), # 11
    ('dhi_6h',True), # 12
    ('dhi_12h',True), # 13
    ('dhi_24h',True), # 14
    ('dni - direct solar irradiance',True), # 15
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
from operator import concat
from re import L
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

    @property 
    def num_agents(self):
        return self.input_feature_engineer.num_agents

    @property
    def shared(self):
        return self.input_feature_engineer.shared
    
    @property
    def private(self):
        return self.input_feature_engineer.personal

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
    
    def register_reset(self, observation):
        self._out = np.array(observation['observation']).shape
        self.observation = observation
        self._num_agents = len(observation['observation'])

    @property 
    def num_agents(self):
        return self._num_agents

    @property
    def out(self):
        return self._out

    @property
    def shared(self):
        return 20

    @property
    def personal(self):
        return 8
        
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
        return (self.shared+self.num_agents*self.private,)
    
    def _transform(self, x):
        return np.concatenate((x[0,:self.shared],x[:,self.shared:].flatten()),axis=None)

import holidays
from datetime import date
class RuleFeatureEngineerV0(FeatureEngineer):
    '''
    Manually Tuned
    '''
    def __init__(self, input_feature_engineer):
        super(RuleFeatureEngineerV0, self).__init__(input_feature_engineer)
        assert(isinstance(input_feature_engineer,BaseFeatureEngineer))
        self.holidays = holidays.US()

    @property
    def out(self,):
        return (self.num_agents,35)

    @property
    def shared(self):
        return 24

    @property
    def personal(self):
        return 11   
    
    def _transform(self,x):

        year = 2015 if x[0][0] >= 8 else 2016
        # Time data 
        d = date(year = year, month = int(x[0][0]), day = int(x[0][1]))
            # Holiday
        if d in self.holidays:
            holiday = 1
        else:
            holiday = 0

            # Weekend
        if d.weekday() > 4:
            weekend = 1
        else:
            weekend = 0 

            # Working hour
        hour = x[0][2]
        if 9 <= hour <= 14 and weekend == 0 and holiday == 0:
            working = 1
        else:
            working = 0

            # Cyclical
        day = d.timetuple().tm_yday
        sin_day = np.sin(day/183)
        cos_day = np.cos(day/183)
        sin_hour = np.sin(hour/12)
        cos_hour = np.cos(hour/12)

        shared_time = np.array([holiday,weekend,working,sin_day,cos_day,sin_hour,cos_hour])
        shared_time = np.broadcast_to(shared_time,(5,7))
        # Weather data
        # TODO: We probably want to make predictors using weather data
        # For solar, carbon and price, then remove original data for dimensionality
        # reduction
            # Normalization
        shared_weather = x[:,3:20]
        mean = np.array([16.84, 16.84, 16.84, 16.84, 73.0, 73.0, 73.0, 73.0, 208.31, 208.3, 208.2, 208.31, 201.25, 201.24, 201.16, 201.25, 0.16])
        std = np.array([3.56, 3.56, 3.56, 3.56, 16.48, 16.48, 16.48, 16.48, 292.81, 292.81, 292.7, 292.81, 296.2, 296.21, 296.14, 296.2, 0.04])

        shared_weather = (shared_weather-mean) / std

        # Private data

        base_consumption = x[:,20]
        solar_generation = x[:,21]
        electrical_storage = x[:,22]
        net_consumption = x[:,23]
        pricing = (x[:,24:] - 0.273) / 0.118

        spare_solar_capacity = np.maximum(0,solar_generation-base_consumption)
        clipped_no_action_consumption = np.maximum(0,base_consumption - solar_generation)
        clipped_net_consumption = np.maximum(0,net_consumption)

        private_observation = np.stack((base_consumption,solar_generation,net_consumption,electrical_storage,spare_solar_capacity,clipped_no_action_consumption, clipped_net_consumption)).T
        private_observation = np.concatenate((private_observation,pricing),axis=1)

        # TODO: Building

        # Result
        result = np.concatenate((shared_time, shared_weather, private_observation),axis=1,dtype=np.float32)

        return result