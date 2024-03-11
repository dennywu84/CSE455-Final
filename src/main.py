import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import loader
import neuralnet
import helper

if __name__ == "__main__":
    # Uses a hardware accelerator such as GPU if available. If not, uses CPU
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

    # testing json_to_csv
    df = loader.json_to_csv('../data/raw_data/public_training_set_release_2.0/annotations.json')
    print(df)

    train_dataloader, valid_dataloader = loader.get_data_loader(32)

    epochs = 10
    for epoch in range(epochs):
        learning_rate = 1e-3
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print(f"Epoch {epoch+1}\n-------------------------------")
        helper.train(train_dataloader, model, loss_fn, optimizer)
        helper.test(valid_dataloader, model, loss_fn)

    
    # print(df['category_id'].nunique())


