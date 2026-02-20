import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')
features = torch.tensor(data.drop('MPG', axis = 1).to_numpy()).float()
print(features)

target = torch.tensor(data['MPG'].to_numpy()).float().reshape(-1,1)
print(target)

fmean = features.mean(axis = 0, keepdim = True)
print(fmean)
fSTD = features.std(axis = 0, keepdim = True)
print(fSTD)

tmean = target.mean(axis = 0)
print(tmean)
tSTD = target.std(axis = 0)
print(tSTD)

X = (features - fmean)/fSTD
print(X)

Y = (target - tmean)/tSTD
print(Y)

print()

model = nn.Linear(2,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = .1)

epochs = 10000
for epoch in range(epochs):
    Yhat = model(X)
    loss = criterion(Yhat,Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

torch.save({
    'fm':fmean,
    'fs':fSTD,
    'tm':tmean,
    'ts':tSTD,
    'parameters':model.state_dict()
},'model.pth')
