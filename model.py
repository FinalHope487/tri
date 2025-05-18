import torch.nn as nn
from torchvision import models

def create_resnet34():
    resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    resnet34.fc = nn.Sequential(
        nn.Linear(resnet34.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
        nn.Sigmoid()
    )
    return resnet34

def create_resnet18(num_classes=3):
    resnet18 = models.resnet18()
    resnet18.fc = nn.Sequential(
        nn.Linear(resnet18.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )
    return resnet18