import torch as t
import torch.nn as nn
import torch.optim as optim
from dataset import SolarPredDataset
from torch.utils.data import DataLoader
from solar_pred_model import SolarPredModel

device = "cuda" if t.cuda.is_available() else "cpu"
print(f"Using {device} device")

test_dataset = SolarPredDataset('build1/test_data1.pkl')
# test_dataset = SolarPredDataset('build2/test_data2.pkl')
# test_dataset = SolarPredDataset('build3/test_data3.pkl')
# test_dataset = SolarPredDataset('build4/test_data4.pkl')
# test_dataset = SolarPredDataset('build5/test_data5.pkl')
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)


model = SolarPredModel().to(device)
criterion = nn.MSELoss()

model.load_state_dict(t.load('solar_pred_model1.pt'))
# model.load_state_dict(t.load('solar_pred_model2.pt'))
# model.load_state_dict(t.load('solar_pred_model3.pt'))
# model.load_state_dict(t.load('solar_pred_model4.pt'))
# model.load_state_dict(t.load('solar_pred_model5.pt'))

model.eval()
total_loss = 0
with t.no_grad():
    for i, (x, y) in enumerate(test_dataloader):
        y_pred = model(x.to(device))
        loss = criterion(y_pred, y.to(device))
        # print('epoch: {}, batch: {}, loss: {}'.format(epoch, i, loss.item()))
        total_loss += loss.item()
print('test loss: {}'.format(total_loss / len(test_dataloader)))

c = model(t.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=t.float32).to(device))
a0 = model(t.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=t.float32).to(device)) - c
a1 = model(t.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=t.float32).to(device)) - c
a2 = model(t.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=t.float32).to(device)) - c
a3 = model(t.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=t.float32).to(device)) - c
a4 = model(t.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=t.float32).to(device)) - c

c = c.detach().item()
a0 = a0.detach().item()
a1 = a1.detach().item()
a2 = a2.detach().item()
a3 = a3.detach().item()
a4 = a4.detach().item()

print(f'solar gen = {c} + {a0} * month + {a1} * temp + {a2} * humid + {a3} * diff + {a4} * direct')