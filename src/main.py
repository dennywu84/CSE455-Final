import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import loader
import neuralnet

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

