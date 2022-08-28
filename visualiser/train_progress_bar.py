import ipywidgets as widgets

class TrainProgressBar:
    def __init__(self):
        self.s = widgets.IntProgress(value=0,min=0,max=365*24)

    def initialize_progress(self,episode):
        self.s.max = 365*24*episode
        self.s.value = 0

    def update_progress_by_one(self):
        self.s.value += 1

    def get_object(self):
        return self.s

    def close(self):
        pass