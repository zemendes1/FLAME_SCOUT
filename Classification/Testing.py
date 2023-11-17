import os

import torch
#  imports for the network
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# import other files
import FLAME_SCOUT_model as FL_Model

# Alter the path to the model accordingly
model_path = "pytorch_model_0.pth"

# Get files in current working directory
files = os.listdir()


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


# Check the dataset and the model are present in the files
if 'Dataset' in files and model_path in files:

    # Make the Dataset and DataLoader for Testing
    Testing_path = 'Dataset/Classification/Test'
    transform = transforms.Compose([transforms.ToTensor()])
    testing_dataset = torchvision.datasets.ImageFolder(Testing_path, transform=transform)
    batch_size_testing = 100
    testing_loader = DataLoader(testing_dataset, batch_size=batch_size_testing, shuffle=True, num_workers=0)

    # Load the model
    model = FL_Model.FLAME_SCOUT_Model(num_classes=2)
    model.load_state_dict(torch.load(model_path))

    # Print Model
    print(model)

    # Do the Testing
    evaluate(model, testing_loader)

# Dataset or Model weren't found
else:
    print("Dataset or Model not found. Please download the dataset and unzip it to the current working directory.")
