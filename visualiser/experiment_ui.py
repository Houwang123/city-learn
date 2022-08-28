import ipywidgets as widgets
import numpy as np
import os
from visualiser.frame_cache import *
from ipywidgets import *
from visualiser.one_tab import OneTab

class ExperimentUI:

    def __init__(self, allow_attributes_configuration=True):
        self.existing_experiment_ids = []
        for root, dirs, files in os.walk('experiments'):
            self.existing_experiment_ids = dirs[:-1]
            break
        self.experiment_tab = widgets.Tab()
        self.experiment_tab.titles = [str(i) for i in range(len(self.existing_experiment_ids))]
        self.experiment_tab.children = [OneTab(int(id),allow_attributes_configuration).get_ui_object() for id in self.existing_experiment_ids]

        def add_tab_to_experiment_tab(c):
            def get_new_experiment_id():
                self.id = 0
                while os.path.exists(os.path.join('experiments',str(self.id))):
                    self.id += 1
                return self.id
            self.new_id = get_new_experiment_id()
            self.new_tab = OneTab(self.new_id)
            self.experiment_tab.children += (self.new_tab.get_ui_object(),)
            self.experiment_tab.titles += (str(self.new_id),)

        self.add_new_tab_button = widgets.Button(description='New Experiment',disabled=False,button_style='',tooltip='Click me',icon='plus')
        self.add_new_tab_button.on_click(add_tab_to_experiment_tab)

        self.experiment_ui = widgets.VBox([self.add_new_tab_button, self.experiment_tab])
        
    def get_ui_object(self):
        return self.experiment_ui