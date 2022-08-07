import enum
import matplotlib.pyplot as plt
import numpy as np

class SupervisorAsVisualiser:
    all_reward_for_the_day = None
    reward_vs_time_data_x = None
    reward_vs_time_data_y = None
    day_count = 0

    @classmethod
    def append_reward(self, rewards_for_one_step):
        rewards_for_one_step = list(rewards_for_one_step)
        # print(rewards_for_one_step)
        if SupervisorAsVisualiser.all_reward_for_the_day is None:
            SupervisorAsVisualiser.all_reward_for_the_day = [[] for _ in rewards_for_one_step]
            if SupervisorAsVisualiser.reward_vs_time_data_x is None:
                SupervisorAsVisualiser.reward_vs_time_data_x = [[] for _ in rewards_for_one_step]
                SupervisorAsVisualiser.reward_vs_time_data_y = [[] for _ in rewards_for_one_step]
        for agent_id, _ in enumerate(rewards_for_one_step):
            SupervisorAsVisualiser.all_reward_for_the_day[agent_id].append(rewards_for_one_step[agent_id])

    @classmethod
    def inform_day_end(self):
        SupervisorAsVisualiser.day_count += 1
        for agent_id, _ in enumerate(SupervisorAsVisualiser.all_reward_for_the_day):
            SupervisorAsVisualiser.reward_vs_time_data_y[agent_id].append( \
                np.average(np.array(SupervisorAsVisualiser.all_reward_for_the_day[agent_id])))
            SupervisorAsVisualiser.reward_vs_time_data_x[agent_id].append(SupervisorAsVisualiser.day_count)
        SupervisorAsVisualiser.all_reward_for_the_day = None

    @classmethod
    def plot_reward_vs_time(self):
        fig, ax = plt.subplots()
        for agent_id, _ in enumerate(SupervisorAsVisualiser.reward_vs_time_data_x):
            ax.plot(SupervisorAsVisualiser.reward_vs_time_data_x[agent_id], SupervisorAsVisualiser.reward_vs_time_data_y[agent_id])

        plt.show()
