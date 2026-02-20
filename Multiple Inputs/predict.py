import torch
import torch.nn as nn

modelData = torch.load('model.pth')
fMean = modelData['fm']
fSTD = modelData['fs']
tMean = modelData['tm']
tSTD = modelData['ts']

parameters = modelData['parameters']
print(parameters)

model = nn.Linear(2,1)
model.load_state_dict(parameters)

features = torch.tensor([
    [3,2]
]).float()
print(features)

X = (features-fMean)/fSTD


prediction = model(X)

output = prediction*tSTD+tMean
print(output)