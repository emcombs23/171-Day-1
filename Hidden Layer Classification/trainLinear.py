import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from export import export_model

data = pd.read_csv('data.csv')
X = torch.tensor(data.drop('y', axis = 1).values).float()
print(X)

Y = torch.tensor(data['y'].values).float().reshape(-1,1)
print(Y)

model = nn.Linear(2,1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = .01)

epochs = 1000

for epoch in range(epochs):
    yHat = model(X)
    loss = criterion(yHat,Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss)

export_model(model, 'model.json')