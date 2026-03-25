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

testData = datasets.MNIST(
    root = './data',
    train = False,
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

testLoader = DataLoader(
    testData,
    batch_size = 1000,
    shuffle = False
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
    total_loss = 0
    correct = 0
    images = 0
    for image,label in loader:
        optimizer.zero_grad()
        X = image
        Y = label
        yHat = model(X)
        loss = criterion(yHat,Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        images += label.size(0)
        correct += sum(yHat.argmax(1) == label).item()
    print(correct, images)
    print(total_loss/len(loader))


