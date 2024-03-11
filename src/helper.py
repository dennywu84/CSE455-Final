import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

import tqdm
import numpy as np



# We can try using SGD, ADAM, or RMSProp optimizers to see if one is better. Default will just be no optimizer
def train(dataloader, model, optimizer=None):
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    total_correct = 0
    size = len(dataloader.dataset)

    for inputs, labels in tqdm.tqdm(dataloader):
        # Predict data (x) and then compare with labels (y)
        prediction = model(inputs)
        loss = loss_fn(prediction, labels)
        
        # Keep track of # of correct predictions
        _, predicted = torch.max(prediction, 1)
        total_correct += (predicted == labels).sum().item()

        # Backpropogation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    accuracy = total_correct / size
    print(f"Accuracy: {accuracy * 100:.2f}%")

def test(dataloader, model):
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    size = len(dataloader.dataset)
    total_loss, correct = 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(dataloader):
            prediction = model(inputs)
            loss = loss_fn(prediction, labels)
            total_loss += loss.item()
            correct += (prediction.argmax(1) == labels).type(torch.float).sum().item()
    
    avg_loss = total_loss/len(dataloader)
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
