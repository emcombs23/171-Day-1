from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from grid import save_image_grid
import torch

data = datasets.MNIST(
    root = './data',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

torch.manual_seed(1)

#image, label = data[0]
#print(image)

loader = DataLoader(
    data,
    batch_size = 64,
    shuffle = True
)

'''
for i, (images, labels) in enumerate(loader):
    save_image_grid(images)
    if i == 9:
        break
'''

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = .001)
epochs = 10
print('Hello')
for epoch in range(epochs):
    for i,(image, label) in enumerate(loader):
        X = image
        Y = label
        yHat = model(X)
        loss = criterion(yHat,Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(loss)
