import torch.nn as nn
from torchvision import models

def create_player():
    resnet18 = models.resnet18()
    resnet18.fc = nn.Sequential(
        nn.Linear(resnet18.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
        nn.Sigmoid()
    )
    return resnet18

def create_classifier(num_classes=3):
    resnet18 = models.resnet18()
    resnet18.fc = nn.Sequential(
        nn.Linear(resnet18.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )
    return resnet18