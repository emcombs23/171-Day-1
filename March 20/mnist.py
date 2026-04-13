from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from grid import save_image_grid
import torch

data = datasets.CIFAR10(
    root = './data',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

testData = datasets.CIFAR10(
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
    nn.Conv2d(3,32,kernel_size = 3, padding = 1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32,64,kernel_size = 3, padding = 1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64*8*8, 128),
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
    print(correct,images, correct/images)
    print(total_loss/len(loader))

    testCorrect = 0
    testTotal = 0
    with torch.no_grad():
        for images, labels in testLoader:
            output = model(images)
            testCorrect += (output.argmax(1) == labels).sum().item()
            testTotal += labels.size(0)
    print(testCorrect, testTotal, testCorrect/testTotal)
    print("------------------")
    
torch.save(model.state_dict(),'cifarModel2.pth')

