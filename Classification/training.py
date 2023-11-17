"""
Make all the imports:
"""

# import for checking the files
import os

#  imports for the network
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# import other files
import FLAME_SCOUT_model as FL_Model

"""Download the Dataset for Classification:"""

# Get files in current working directory
files = os.listdir()

if 'Dataset' not in files:
    raise Exception("Dataset not found. Please download the dataset and unzip it to the current working directory.")

"""Preprocessing and examination:"""

# Get the current folder paths of the dataset
Training_path = 'Dataset/Classification/Training'
Testing_path = 'Dataset/Classification/Test'

# Create the datasets
transform = transforms.Compose([transforms.ToTensor()])
training_dataset = torchvision.datasets.ImageFolder(Training_path, transform=transform)
testing_dataset = torchvision.datasets.ImageFolder(Testing_path, transform=transform)

# Split the training data (80%)
train_len = len(training_dataset.samples)
val_split = 0.2
val_len = int(val_split * train_len)
train_len -= val_len
train_dataset, validation_dataset = torch.utils.data.random_split(training_dataset, [train_len, val_len])

# Create the dataloaders
batch_size_training = 100
training_loader = DataLoader(training_dataset, batch_size=batch_size_training, shuffle=True, num_workers=0)

batch_size_validation = 100
validation_loader = DataLoader(validation_dataset, batch_size=batch_size_training, shuffle=True, num_workers=0)

batch_size_testing = 100
testing_loader = DataLoader(testing_dataset, batch_size=batch_size_testing, shuffle=True, num_workers=0)

print('Length of the Training set: ' + str(train_len))
print('Length of the Validation set: ' + str(len(testing_dataset.samples)))
print('Length of the Testing set: ' + str(len(testing_dataset.samples)))

class_map = {0: 'Fire', 1: 'No Fire'}

# Print one image
dataiter = iter(training_loader)
images, labels = next(dataiter)

for image, label in zip(images, labels):
    plt.figure()
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.title(class_map[label.item()])
    break

# Create an instance of the model
num_classes = 2
input_shape = (3, 254, 254)
model = FL_Model.FLAME_SCOUT_Model(num_classes)

# Print the model architecture
print(model)


def train_pytorch_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, save_model_flag=False):
    for epoch in range(epochs):

        # Initialize Accuracy Values
        training_loss = 0
        correct = 0
        total = 0

        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        training_loss /= len(train_loader)
        training_accuracy = correct / total

        model.eval()

        # Save the model after training 1 epoch
        if save_model_flag:
            torch.save(model.state_dict(), "pytorch_model_{}.pth".format(epoch))

        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        accuracy = correct / total

        print(
            f"Epoch [{epoch + 1}/{epochs}]- Training Loss: {training_loss:.4f} - Training Accuracy: {100 * training_accuracy:.2f} - Validation Loss: {val_loss:.4f} - Validation Accuracy: {100 * accuracy:.2f}%")


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define number of epochs
epochs = 10

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_pytorch_model(model, training_loader, validation_loader, criterion, optimizer, epochs, device, save_model_flag=True)
