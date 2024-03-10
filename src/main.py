import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import loader
import neuralnet

if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = neuralnet.NeuralNetwork().to(device)
    print(model)

    df = loader.json_to_csv('../data/raw_data/public_validation_set_2.0/annotations.json', '../data/raw_data/public_validation_set_2.0/images')
    print(df)


