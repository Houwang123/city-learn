import numpy as np


def rbc_policy(observation, action_space, agent_id):
    month, day, hour = observation[0], observation[1], observation[2]
    outdoor_temp, outdoor_temp_6h, outdoor_temp_12h, outdoor_temp_24h = observation[3], observation[4], observation[5], \
                                                                        observation[6]
    outdoor_humidity, outdoor_humidity_6h, outdoor_humidity_12h, outdoor_humidity_24h = observation[7], observation[8], \
                                                                                        observation[9], observation[10]
    diffuse_solar_radiation, diffuse_solar_radiation_6h, diffuse_solar_radiation_12h, diffuse_solar_radiation_24h = \
        observation[11], observation[12], observation[13], observation[14]
    direct_solar_radiation, direct_solar_radiation_6h, direct_solar_radiation_12h, direct_solar_radiation_24h = \
        observation[15], observation[16], observation[17], observation[18]
    carbon_intensity = observation[19]
    non_shiftable_load = observation[20]
    solar_generation = observation[21]
    electrical_storage_soc = observation[22]
    net_consumption = observation[23]
    electricity_pricing, electricity_pricing_6h, electricity_pricing_12h, electricity_pricing_24h = \
        observation[24], observation[25], observation[26], observation[27]

    assert len(observation) == 28

    action = 1
    # if solar_generation > non_shiftable_load:
    #     action = 0.01 #(solar_generation - non_shiftable_load) / 10
    # else:
    #     action = -0.01 #min(electrical_storage_soc, (non_shiftable_load - solar_generation) / 10)

    if agent_id == 0:
        # print net consumption, soc, load, solar generation
        print(
            f"Net consumption: %.2f \t SOC: %.2f \t Load: %.2f \t Solar generation: %.2f" % (
                net_consumption, electrical_storage_soc, non_shiftable_load, solar_generation))


    # if electrical_storage_soc <= 0.5:
    #     action = 1
    # else:
    #     action = -1

    action = np.array([action], dtype=action_space.dtype)
    assert action_space.contains(action)
    return action


class MyAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """

    def __init__(self):
        self.action_space = {}

    def register_reset(self, observation, action_space, agent_id):
        """Get the first observation after env.reset, return action"""
        self.action_space[agent_id] = action_space
        return rbc_policy(observation, action_space, agent_id)

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        return rbc_policy(observation, self.action_space[agent_id], agent_id)
