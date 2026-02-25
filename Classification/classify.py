import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as numpy

torch.manual_seed(42)

data = pd.read_csv('data.csv')
data['Diagnosis'] = data['Diagnosis'].map({'Benign':0,'Malignant':1})

feature = torch.tensor(data.drop('Diagnosis', axis = 1).to_numpy()).float()
target = torch.tensor(data['Diagnosis'].to_numpy()).float().reshape(-1,1)


fMean = feature.mean(axis = 0,keepdim = True)
fSTD = feature.std(axis = 0,keepdim = True)


X = (feature-fMean)/fSTD
Y = target

model = nn.Linear(1,1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr = .1)
epochs = 250

for epoch in range(epochs):
    yHat = model(X)
    loss = criterion(yHat, Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(loss)


torch.save({
    'fm': fMean,
    'fs': fSTD,
    'parameters': model.state_dict()
}, 'model.pth')