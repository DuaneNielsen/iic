from torchvision import transforms as T

celeba_transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
])

