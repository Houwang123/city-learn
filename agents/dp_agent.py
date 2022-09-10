import pickle


class DPAgent:
    def __init__(self):
        self.action_space = {}
        self.last_action = None
        self.episode_counter = 0
        self.step_counter = 0
        self.actions = dict()
        for agent_id in range(5):
            with open(f'./actions/{agent_id}_actions.pkl', 'rb') as file:
                actions = pickle.load(file)
                self.actions[agent_id] = actions

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        if agent_id == 0:
            self.episode_counter += 1
            self.step_counter = 0

    def compute_action(self, observation, agent_id):
        if agent_id == 0:
            self.step_counter += 1

        try:
            action = self.actions[agent_id][self.step_counter - 1]
        except IndexError:
            action = 0

        action = [action]
        return action
