import torch as t
import torch.nn as nn


class SolarPredModel(nn.Module):
    def __init__(self):
        super(SolarPredModel, self).__init__()
        self.fc1 = nn.Linear(5, 1000)
        # self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = t.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = t.relu(x)
        x = self.bn2(x)
        x = self.fc3(x)
        return x.squeeze(1)
