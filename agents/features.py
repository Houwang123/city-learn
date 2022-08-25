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
import numpy as np
import torch
from citylearn import preprocessing
# Example usage 
# engineer = DecoratorA(DecoratorB(BaseFeatureEngineer(observations)))
# shape = engineer.out
# result = engineer.tranform(obs)

class FeatureEngineer(ABC):

    def __init__(self,input_feature_engineer, central_agent: bool):
        self.input_feature_engineer = input_feature_engineer
        
        self.central_agent = central_agent
        
        self.shared_keys = ['month', 'day_type', 'hour',
                            'outdoor_dry_bulb_temperature', 'outdoor_dry_bulb_temperature_predicted_6h',
                            'outdoor_dry_bulb_temperature_predicted_12h', 'outdoor_dry_bulb_temperature_predicted_24h',
                            'outdoor_relative_humidity', 'outdoor_relative_humidity_predicted_6h',
                            'outdoor_relative_humidity_predicted_12h', 'outdoor_relative_humidity_predicted_24h',
                            'diffuse_solar_irradiance', 'diffuse_solar_irradiance_predicted_6h',
                            'diffuse_solar_irradiance_predicted_12h', 'diffuse_solar_irradiance_predicted_24h',
                            'direct_solar_irradiance', 'direct_solar_irradiance_predicted_6h',
                            'direct_solar_irradiance_predicted_12h', 'direct_solar_irradiance_predicted_24h',
                            'carbon_intensity']
        
        self.internal_keys = ['non_shiftable_load', 'solar_generation',
                                'electrical_storage_soc', 'net_electricity_consumption',
                                'electricity_pricing', 'electricity_pricing_predicted_6h',
                                'electricity_pricing_predicted_12h', 'electricity_pricing_predicted_24h']
        
        self.output = None

    @property
    @abstractmethod
    def out(self):
        # Should be either be constant or dependent on out of input
        pass

    @abstractmethod 
    def _transform(self, *args):
        if self.central_agent:
            shared_obs = {k: v for k, v in zip(self.shared_keys, observations)}
        else:
            shared_obs = {k: v for k, v in zip(self.shared_keys, observations[0])}
            
        shared_obs_norm = shared_obs.copy()

        for names, values in shared_obs.items():
          if names in ['month']:
            shared_obs_norm[names] = preprocessing.PeriodicNormalization(12) * shared_obs[names]
          elif names in ['day_type']:
            shared_obs_norm[names] = preprocessing.OnehotEncoding([1, 2]) * (
            1 if shared_obs[names] <= 5 else 2)
          elif names in ['hour']:
            shared_obs_norm[names] = preprocessing.PeriodicNormalization(24) * shared_obs[names]
          else:
            shared_obs_norm[names] = non_zero_normalize(min(shared_obs[names]), max(shared_obs[names])) * shared_obs[names]
            
          return shared_obs, shared_obs_norm
    
    def _calculate_internal_values(shared_obs):
         '''
        may be for future envviroment model
        '''
        pass

    def transform(self, x, *args):
        '''
        Transforms a single observation 
        '''
        shared_obs, shared_obs_norm = self._transform(observations)
        #internal_obs = self._calculate_internal_values(shared_obs)


        obs_shared = np.array([shared_obs[keys] for keys in self.shared_keys])

        pca = PCA(n_components = n_components_shared)
        obs_shared = pca.fit_transform(shared_obs)

        return obs_shared

    def batch_transform(self, x, *args):
        '''
        Transform but with an extra first dimension.
        Can overwrite to keep vectorization performance boost.
        '''
        outs = []
        for i in range(len(x)):
            outs.append(self.transform(x[i]))
        
        return outs

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
    
class non_zero_normalize:
  def __init__(self, x_min, x_max):
    self.x_min = x_min
    self.x_max = x_max

  def __mul__(self, x):
    if self.x_min == self.x_max:
      return 0
    elif min(x) != 0 or max(x) == 0:
      return (x - self.x_min)/(self.x_max - self.x_min)
    else:
      x_non_zero = x[x>0]
      x_idx = x.copy()
      x_idx[x!=0] = 1
      return (0.1 + 0.9*(x - min(x_non_zero))/(max(x_non_zero) - min(x_non_zero)))*x_idx
  def __rmul__(self, x):
    if self.x_min == self.x_max:
      return 0
    elif min(x) != 0 or max(x) == 0:
      return (x - self.x_min)/(self.x_max - self.x_min)
    else:
      x_non_zero = x[x>0]
      x_idx = x.copy()
      x_idx[x!=0] = 1
      return (0.1 + 0.9*(x - min(x_non_zero))/(max(x_non_zero) - min(x_non_zero)))*x_idx
    
