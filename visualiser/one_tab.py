from re import T
import ipywidgets as widgets
import threading
import numpy as np
import os
from visualiser.frame_cache import *
from ipywidgets import *
from visualiser.train_progress_bar import TrainProgressBar
from train import train

class OneTab():

    def __init__(self, experiment_id, allow_attributes_configuration=True):

        self.frame_cache = FrameCache()
        self.train_progress_bar_o = TrainProgressBar()
        self.training_thread = threading.Thread(target=train, args=('visualiser/created_experiment.json',True,self.train_progress_bar_o, self.frame_cache))
        self.current_id = experiment_id

        self.train_button = widgets.Button(layout=Layout(width='800px'),description='Start Training',disabled=False,button_style='')
        self.img_ui = widgets.VBox()

        def wait_for_train_finished():
            self.training_thread.join()
            self.train_button.disabled = False
            self.training_thread = threading.Thread(target=train, args=('visualiser/created_experiment.json',True,self.train_progress_bar_o, self.frame_cache))
            self.wait_for_training_thread = threading.Thread(target=wait_for_train_finished)
            self.train_button.description = 'Restart Training'
            from IPython.display import display
            def _show(x):
                display(self.frame_cache.get_image_of_frame_at(x))
            self.frame_slider = widgets.IntSlider(min=0, max=self.frame_cache.get_total_frame_number() - 1, value=0)
            self.frame_player = widgets.Play(
                value=0,
                min=0,
                max=self.frame_cache.get_total_frame_number() - 1,
                step=1,
                interval=500,
                description="Press play",
                disabled=False
            )
            self.frame_input_ui = widgets.HBox([self.frame_player, self.frame_slider])
            widgets.jslink((self.frame_slider, 'value'), (self.frame_player, 'value'))
            self.img_output = widgets.interactive_output(_show, {'x': self.frame_slider})
            self.img_ui.children = [self.frame_input_ui, self.img_output]
            
        self.wait_for_training_thread = threading.Thread(target=wait_for_train_finished)

        def train_button_clicked(c):
            self.train_button.disabled = True
            set_json_file()
            self.train_progress_bar_o.initialize_progress(self.total_episode_textbox.value)
            self.training_thread.start()
            self.wait_for_training_thread.start()

        self.train_button.on_click(train_button_clicked)
        from rewards import rewards
        self.reward_function_vars = np.array(dir(rewards))
        self.reward_function_names = self.reward_function_vars[np.char.endswith(self.reward_function_vars,"reward")]
        import sys
        self.agent_type_vars = np.array([])
        for root, dirs, files in os.walk('agents/agents'):
            for file in files:
                if file.endswith('.py') and not(file.endswith('__.py')):
                    self.module = sys.modules[(root + '/' + file[0:-3]).replace('/','.')]
                    self.agent_type_vars = np.append(self.agent_type_vars, np.array(dir(self.module)))
        self.agent_type_names = self.agent_type_vars[np.char.endswith(self.agent_type_vars,"Agent")]
        self.total_episode_textbox = widgets.BoundedIntText(layout=Layout(width='800px'),value=1,min=1,max=999,step=1,description='Number of episodes (years):',style={'description_width':'initial'})
        self.steps_per_frame_save_textbox = widgets.BoundedIntText(layout=Layout(width='800px'),value=200,min=1,max=365*24-1,step=1,description='Steps per frame save:',style={'description_width':'initial'})
        self.reward_function_dropdown = widgets.Dropdown(layout=Layout(width='800px'),options=self.reward_function_names, value='simple_reward',description='Reward function (has to end with reward):',style={'description_width':'initial'})
        self.agent_type_dropdown = widgets.Dropdown(layout=Layout(width='800px'),options=self.agent_type_names, value='TD3Agent',description='Agent (has to end with Agent):',style={'description_width':'initial'})
        self.agent_attributes_textbox = widgets.Textarea(layout=Layout(width='800px',height='450px'),style={'description_width':'initial'},value='\"actor\": \"CommNet\",\n\"critic\": \"CentralCritic\",\n\"actor_feature\": \"RuleFeatureEngineerV0(BaseFeatureEngineer())\",\n\"critic_feature\": \"CentralCriticEngineer(RuleFeatureEngineerV0(BaseFeatureEngineer()))\", \n\"a_kwargs\": {\n\"comm_steps\" : 5,\n\"comm_size\":6,\n\"hidden_size\": 64\n},\n\"c_kwargs\": {\n\"hidden_size\": 128\n},\n\"gamma\": 0.99, \n\"lr\":3e-4,\n\"tau\":0.001,\n\"batch_size\":256,\n\"memory_size\":65536,\n\"device\": \"\'cpu\'\"')
        # agent_attributes_textbox = widgets.Label(style={'description_width':'initial'},value='\"actor\": \"CommNet\",\n\"critic\": \"CentralCritic\",\n\"actor_feature\": \"RuleFeatureEngineerV0(BaseFeatureEngineer())\",\n\"critic_feature\": \"CentralCriticEngineer(RuleFeatureEngineerV0(BaseFeatureEngineer()))\", \n\"a_kwargs\": {\n\"comm_steps\" : 5,\n\"comm_size\":6,\n\"hidden_size\": 64\n},\n\"c_kwargs\": {\n\"hidden_size\": 128\n},\n\"gamma\": 0.99, \n\"lr\":3e-4,\n\"tau\":0.001,\n\"batch_size\":256,\n\"memory_size\":65536,\n\"device\": \"\'cpu\'\"')
        # agent_attributes_textbox = widgets.Label(value="The $m$ in $E=mc^2$:")
        import json
        def set_json_file():
            with open('visualiser/created_experiment.json', 'r') as f:
                self.json_data = json.load(f)
                self.json_data['id'] = self.current_id
                self.json_data['episodes'] = self.total_episode_textbox.value
                self.json_data['steps_per_frame_save'] = self.steps_per_frame_save_textbox.value
                self.json_data['reward'] = self.reward_function_dropdown.value
                self.json_data['agent']['type'] = self.agent_type_dropdown.value
                self.json_data['agent']['attributes'] = json.loads('{' + self.agent_attributes_textbox.value + '}')
            with open('visualiser/created_experiment.json', 'w') as f:
                f.write(json.dumps(self.json_data))

        self.train_progress_bar = self.train_progress_bar_o.get_object()
        self.train_progress_bar_o.initialize_progress(self.total_episode_textbox.value)
        self.experiment_input = widgets.VBox([self.total_episode_textbox, self.steps_per_frame_save_textbox, \
            self.reward_function_dropdown, self.agent_type_dropdown])
        if allow_attributes_configuration:
            self.experiment_input = widgets.VBox([self.total_episode_textbox, self.steps_per_frame_save_textbox, \
            self.reward_function_dropdown, self.agent_type_dropdown, self.agent_attributes_textbox])
        self.experiment_ui = widgets.VBox([self.experiment_input, self.train_button, self.train_progress_bar])
        self.tab_ui = widgets.VBox([self.experiment_ui, self.img_ui])

    def get_ui_object(self):
        return self.tab_ui