import torch as t
import torch.nn as nn
import torch.optim as optim
from dataset import SolarPredDataset
from torch.utils.data import DataLoader
from solar_pred_model import SolarPredModel

device = "cuda" if t.cuda.is_available() else "cpu"
print(f"Using {device} device")

train_dataset = SolarPredDataset('build1/train_data1.pkl')
# train_dataset = SolarPredDataset('build2/train_data2.pkl')
# train_dataset = SolarPredDataset('build3/train_data3.pkl')
# train_dataset = SolarPredDataset('build4/train_data4.pkl')
# train_dataset = SolarPredDataset('build5/train_data5.pkl')
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)

valid_dataset = SolarPredDataset('build1/valid_data1.pkl')
# valid_dataset = SolarPredDataset('build2/valid_data2.pkl')
# valid_dataset = SolarPredDataset('build3/valid_data3.pkl')
# valid_dataset = SolarPredDataset('build4/valid_data4.pkl')
# valid_dataset = SolarPredDataset('build5/valid_data5.pkl')
valid_dataloader = DataLoader(valid_dataset, batch_size=256, shuffle=True)

# model = SolarPredModel()
model = SolarPredModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(100):
    if epoch % 10 == 0:
        print(f'Epoch {epoch}')
    model.train()
    total_loss = 0
    for i, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        y_pred = model(x.to(device))
        loss = criterion(y_pred, y.to(device))
        loss.backward()
        optimizer.step()
        # print('epoch: {}, batch: {}, loss: {}'.format(epoch, i, loss.item()))
        total_loss += loss.item()
    print('train loss: {}'.format(total_loss / len(train_dataloader)))

    model.eval()
    total_loss = 0
    with t.no_grad():
        for i, (x, y) in enumerate(valid_dataloader):
            y_pred = model(x.to(device))
            loss = criterion(y_pred, y.to(device))
            # print('epoch: {}, batch: {}, loss: {}'.format(epoch, i, loss.item()))
            total_loss += loss.item()
    print('valid loss: {}'.format(total_loss / len(valid_dataloader)))

t.save(model.state_dict(), 'solar_pred_model1.pt')
# t.save(model.state_dict(), 'solar_pred_model2.pt')
# t.save(model.state_dict(), 'solar_pred_model3.pt')
# t.save(model.state_dict(), 'solar_pred_model4.pt')
# t.save(model.state_dict(), 'solar_pred_model5.pt')
