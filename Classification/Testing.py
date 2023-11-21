import os

import torch
#  imports for the network
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
from sklearn.metrics import confusion_matrix

# import other files
import FLAME_SCOUT_model as FL_Model

def evaluate(model, test_loader, epoch):
    model.eval()
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nEpoch {} Test Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, correct, len(test_loader.dataset), accuracy))

    return all_preds, all_targets

# Get files in current working directory
files = os.listdir()

# Number of epochs
epochs = 10

# Check the dataset and the model are present in the files
if 'Dataset' in files in files:

    # Make the Dataset and DataLoader for Testing
    Testing_path = 'Dataset/Classification/Test'
    transform = transforms.Compose([transforms.ToTensor()])
    testing_dataset = torchvision.datasets.ImageFolder(Testing_path, transform=transform)
    batch_size_testing = 100
    testing_loader = DataLoader(testing_dataset, batch_size=batch_size_testing, shuffle=True, num_workers=0)

    for i in range(0, epochs):
        model_path = "pytorch_model_{}.pth".format(i)
            
        if model_path in files :
            # Load the model
            model = FL_Model.FLAME_SCOUT_Model(num_classes=2)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # If the model was saved with DataParallel, remove the 'module' prefix from keys
            state_dict = torch.load(model_path, map_location=device)
            if 'module' in list(state_dict.keys())[0]:
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            model.load_state_dict(state_dict)

            # Do the Testing
            all_targets, all_preds= evaluate(model, testing_loader, i)

    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fire", "No Fire"], yticklabels=["Fire", "No Fire"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()    

# Dataset or Model weren't found
else:
    print("Dataset or Model not found. Please download the dataset and unzip it to the current working directory.")