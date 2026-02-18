import torch
import torch.nn as nn

modelData = torch.load('model.pth')

featureMean = modelData['fm']
featureSTD = modelData['fs']
targetMean = modelData['tm']
targetSTD = modelData['ts']

parameters = modelData['parameters']

features = torch.tensor([
    [1500.0]
])

X = (features-featureMean)/featureSTD

model = nn.Linear(1,1)
model.load_state_dict(parameters)

prediction = model(X)

result = prediction*targetSTD+targetMean
print(result)