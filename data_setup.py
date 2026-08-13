# Imports
import torch
from torchvision import transforms, datasets
import os
from torch.utils.data import dataloader
import pathlib

# Global Variables
BATCH_SIZE = 128
NUM_WORKERS = 0


# Device agnostic code
device = "mps" if torch.mps.is_available() else "cpu"

# Defining the directories
train_dir = pathlib.Path("data/train")
test_dir = pathlib.Path("data/test")

# Visualize one image
train_image_paths = list(train_dir.glob("*/*.jpg"))
test_image_paths = list(test_dir.glob("*/*.jpg"))

# Make the transform for the model
train_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.TrivialAugmentWide(num_magnitude_bins = 31),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Create datasets
train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)

class_names = train_data.classes
class_to_idx = train_data.class_to_idx

# Create dataloaders
train_dataloader = dataloader.DataLoader(train_data,
                                         batch_size = BATCH_SIZE,
                                         num_workers = NUM_WORKERS,
                                         shuffle = True)
test_dataloader = dataloader.DataLoader(test_data,
                                        batch_size = BATCH_SIZE,
                                        num_workers = NUM_WORKERS,
                                        shuffle = False)










