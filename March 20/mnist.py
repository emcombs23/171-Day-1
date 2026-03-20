from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from grid import save_image_grid

data = datasets.MNIST(
    root = './data',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)


#image, label = data[0]
#print(image)

loader = DataLoader(
    data,
    batch_size = 10
)

for i, (images, labels) in enumerate(loader):
    save_image_grid(images)
    if i == 9:
        break