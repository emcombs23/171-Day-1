import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from export import export_model
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)

data = pd.read_csv ('data.csv')
X = torch.tensor(data.drop('y', axis = 1).values).float()
#print(X)

Y = torch.tensor(data['y'].values).float().reshape(-1,1)
#print(Y)


class MyData(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return self.X.size()[0]
    def __getItem__(self, idx):
        return self.X[idx], self.Y[idx]

dataset = MyData(X,Y)
print(len(dataset))

loader = DataLoader(
    dataset,
    batch_size = 10
)
for x,y in loader:
    print(x,y)

model = nn.Linear(2,1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = .01)

epochs = 100

for epoch in epochs():
    for X,Y in loader:
        optimizer.zero_grad()
        yHat = model(X)
        loss = criterion(yHat, Y)
        loss.backward()
        optimizer.step()
    print(loss)