from torchvision import models
from data_setup import train_dataloader, test_dataloader, device
import torch.nn as nn

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights).to(device)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(in_features=model.fc.in_features, out_features=7).to(device)

for param in model.layer3.parameters():
    param.requires_grad = True
    
for param in model.layer4.parameters():
    param.requires_grad = True


