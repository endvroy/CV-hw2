import torch
from torchvision import datasets, transforms

import os

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


def load_train(data_path, batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(data_path, 'train_images'),
                             transform=transform),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1)
    return train_loader


def load_val(data_path, batch_size):
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(data_path, 'val_images'),
                             transform=transform),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=1)
    return val_loader
