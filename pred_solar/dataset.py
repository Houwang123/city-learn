import torch as t
from torch.utils.data import Dataset
import pickle

class SolarPredDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        data = self.data[index]
        x = t.tensor(data[:4], dtype=t.float32)
        y = t.tensor(data[4], dtype=t.float32)
        return x, y

    def __len__(self):
        return len(self.data)