import torch as t
import torch.nn as nn
import torch.optim as optim
from dataset import SolarPredDataset
from torch.utils.data import DataLoader
from solar_pred_model import SolarPredModel

test_dataset = SolarPredDataset('test_data.pkl')
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)


model = SolarPredModel()
criterion = nn.MSELoss()

model.load_state_dict(t.load('solar_pred_model.pt'))

model.eval()
total_loss = 0
with t.no_grad():
    for i, (x, y) in enumerate(test_dataloader):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        # print('epoch: {}, batch: {}, loss: {}'.format(epoch, i, loss.item()))
        total_loss += loss.item()
print('test loss: {}'.format(total_loss / len(test_dataloader)))

c = model(t.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=t.float32))
a0 = model(t.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=t.float32)) - c
a1 = model(t.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=t.float32)) - c
a2 = model(t.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=t.float32)) - c
a3 = model(t.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=t.float32)) - c

c = c.detach().item()
a0 = a0.detach().item()
a1 = a1.detach().item()
a2 = a2.detach().item()
a3 = a3.detach().item()

print(f'solar gen = {c} + {a0} * temp + {a1} * humid + {a2} * diff + {a3} * direct')