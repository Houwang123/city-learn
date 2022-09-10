def random_policy(observation, action_space):
    return action_space.sample()


class MemoryAgent:

    def __init__(self):
        self.action_space = {}
        self.last_action = None
        self.episode_counter = 0
        self.step_counter = 0
        self.memory = dict() # [[] for _ in range(5)] # {dates: observation}


    def set_action_space(self, agent_id, action_space):
        print('agent_id:', agent_id)
        self.action_space[agent_id] = action_space
        if agent_id not in self.memory:
            self.memory[agent_id] = []
        if agent_id == 4:
            self.episode_counter += 1
            self.step_counter = 0
        # if agent_id == 0 and self.episode_counter == 2:
        #     import pickle, sys
        #     with open('observation.pkl', 'wb') as f:
        #         pickle.dump(self.memory, f)
        #     sys.exit(0)


    def compute_action(self, observation, agent_id):
        return [0.]
        """Get observation return action"""
        month, day, hour = observation[0], observation[1], observation[2]
        outdoor_temp, outdoor_temp_6h, outdoor_temp_12h, outdoor_temp_24h = \
            observation[3], observation[4], observation[5], observation[6]
        outdoor_humidity, outdoor_humidity_6h, outdoor_humidity_12h, outdoor_humidity_24h = \
            observation[7], observation[8], observation[9], observation[10]
        diffuse_solar_radiation, diffuse_solar_radiation_6h, diffuse_solar_radiation_12h, diffuse_solar_radiation_24h = \
            observation[11], observation[12], observation[13], observation[14]
        direct_solar_radiation, direct_solar_radiation_6h, direct_solar_radiation_12h, direct_solar_radiation_24h = \
            observation[15], observation[16], observation[17], observation[18]
        unit_carbon_intensity = observation[19]
        non_shiftable_load = observation[20]
        solar_generation = observation[21]
        electrical_storage_soc = observation[22]
        net_consumption = observation[23]
        electricity_pricing, electricity_pricing_6h, electricity_pricing_12h, electricity_pricing_24h = \
            observation[24], observation[25], observation[26], observation[27]



        # print(f'this solar, load, net, soc: {solar_generation}, {non_shiftable_load} {net_consumption} {electrical_storage_soc}')
        mem = observation
        if self.episode_counter == 1:
            self.memory[agent_id].append(mem)
            action = 0
        else:
            if agent_id == -1:
                action = self.agent0_action[self.step_counter]
            else:
                action = 0
                try:
                    next_step = self.memory[agent_id][self.step_counter + 1]
                    next_solar = next_step[21]
                    next_non_shiftable_load = next_step[20]
                    # print(f'predicted solar, load: {next_solar}, {next_non_shiftable_load}')
                    if next_solar > next_non_shiftable_load:
                        action = (next_solar-next_non_shiftable_load)/6.4
                        # print(f'charging. solar: {next_solar}, non_shiftable_load: {next_non_shiftable_load}, '
                        #       f'diff: {next_solar-next_non_shiftable_load}, action: {action}')

                    else:
                        action = (next_solar - next_non_shiftable_load)/6.4
                except IndexError:
                    action = 0
                    print('Index Err, 0')
        if agent_id == 0:
            self.step_counter += 1



                # if mem != past_memory:
                #     print('Wrong prediction at line {} for offset {}'.format(self.episode_counter, (month, day, hour)))
                #     print('Predicted: {}, true: {}'.format(mem, past_memory))
                #     print('\n')



        # action = [1] #random_policy(observation, self.action_space[agent_id])
        action = [action]
        return action

