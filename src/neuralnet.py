import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.conv_relu_stack = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(29*29*32, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 16)

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.conv_relu_stack(x)

        # x = self.conv_relu_stack(x)
        batch_size = logits.shape[0]
        logits = logits.view(batch_size, - 1)
        logits = self.fc1(logits)
        logits = self.dropout(logits)
        logits = self.fc2(logits)
        return logits