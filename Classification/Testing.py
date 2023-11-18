import os

import torch
#  imports for the network
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

# import other files
import FLAME_SCOUT_model as FL_Model

def evaluate(test_model, test_loader):
    test_model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = test_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

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
                model.load_state_dict(torch.load(model_path, map_location=device))    

                # Do the Testing
                evaluate(model, testing_loader)

# Dataset or Model weren't found
else:
    print("Dataset or Model not found. Please download the dataset and unzip it to the current working directory.")
