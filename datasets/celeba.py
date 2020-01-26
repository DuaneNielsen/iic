from torchvision import transforms

celeba_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])