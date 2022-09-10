import torch as t
import torch.nn as nn
import torch.optim as optim
from dataset import SolarPredDataset
from torch.utils.data import DataLoader
from solar_pred_model import SolarPredModel

train_dataset = SolarPredDataset('train_data.pkl')
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)

valid_dataset = SolarPredDataset('valid_data.pkl')
valid_dataloader = DataLoader(valid_dataset, batch_size=256, shuffle=True)

model = SolarPredModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(100):
    print(f'Epoch {epoch}')
    model.train()
    total_loss = 0
    for i, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        # print('epoch: {}, batch: {}, loss: {}'.format(epoch, i, loss.item()))
        total_loss += loss.item()
    print('train loss: {}'.format(total_loss / len(train_dataloader)))

    model.eval()
    total_loss = 0
    with t.no_grad():
        for i, (x, y) in enumerate(valid_dataloader):
            y_pred = model(x)
            loss = criterion(y_pred, y)
            # print('epoch: {}, batch: {}, loss: {}'.format(epoch, i, loss.item()))
            total_loss += loss.item()
    print('valid loss: {}'.format(total_loss / len(valid_dataloader)))

t.save(model.state_dict(), 'solar_pred_model.pt')