import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

import tqdm
import numpy as np

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import constants


# We can try using SGD, ADAM, or RMSProp optimizers to see if one is better
def train(dataloader, model, loss_fn, optimizer, device):

    model.train()
    total_correct = 0
    size = len(dataloader.dataset)

    for inputs, labels in tqdm.tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

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

def test(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    total_loss, correct = 0, 0

    actual_labels = []
    predictions = []
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            prediction = model(inputs)
            total_loss += loss_fn(prediction, labels).item()
            correct += (prediction.argmax(1) == labels).type(torch.float).sum().item()
            
            actual_labels += labels.view(-1).cpu().numpy().tolist()
            _, pred = torch.max(prediction, dim=1)

            predictions += pred.view(-1).cpu().numpy().tolist()

    generate_confusion_matrix(actual_labels, predictions)
    avg_loss = total_loss/len(dataloader)
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {avg_loss:>8f} \n")

def generate_confusion_matrix(actuals, predictions):
    cat_range = constants.MAX_INDEX - constants.MIN_INDEX
    array = [[0] * (cat_range) for _ in range(cat_range)]
    
    assert(len(actuals) == len(predictions))
    for idx in range(len(actuals)):
        array[actuals[idx] - constants.MIN_INDEX][predictions[idx] - constants.MIN_INDEX] += 1

    df_cm = pd.DataFrame(array, range(cat_range), range(cat_range))
    sn.set_theme(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

    plt.savefig('confusion_matrix.png')
    plt.close()