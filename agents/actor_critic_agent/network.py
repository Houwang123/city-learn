import torch as t
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(28, 128)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, 128)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(128, 128)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 21)

    def forward(self, x):
        x = t.relu(self.fc1(x))
        x = self.dropout1(x)
        x = t.relu(self.fc2(x))
        x = self.dropout2(x)
        x = t.relu(self.fc3(x))
        x = self.dropout3(x)
        x = t.relu(self.fc4(x))
        x = self.dropout4(x)
        x = t.relu(self.fc5(x))
        x = self.fc6(x)

        return t.softmax(x, dim=0)

