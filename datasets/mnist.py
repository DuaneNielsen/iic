import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.RandomRotation((-36.0, 36.0), fill=(0,)),
    transforms.RandomResizedCrop((28, 28), scale=(0.85, 1.15), ratio=(3.0/40, 4.0/3.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
