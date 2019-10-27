import torch
from torchvision import datasets
import os


def load_train(data_path, batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(data_path, 'train_images')),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1)
    return train_loader


def load_val(data_path, batch_size):
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(data_path, 'val_images')),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1)
    return val_loader
