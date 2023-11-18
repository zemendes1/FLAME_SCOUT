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

import time

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
training_len = len(training_dataset.samples)
val_split = 0.2
val_len = int(val_split * training_len)
training_len -= val_len
training_dataset, validation_dataset = torch.utils.data.random_split(training_dataset, [training_len, val_len])

# Create the dataloaders
batch_size_training = 32
training_loader = DataLoader(training_dataset, batch_size=batch_size_training, shuffle=True, num_workers=0)

batch_size_validation = 32
validation_loader = DataLoader(validation_dataset, batch_size=batch_size_training, shuffle=True, num_workers=0)

batch_size_testing = 32
testing_loader = DataLoader(testing_dataset, batch_size=batch_size_testing, shuffle=True, num_workers=0)

print('Length of the Training set: '+str(len(training_dataset)))
print('Length of the Validation set: '+str(len(validation_dataset)))
print('Length of the Testing set: '+str(len(testing_dataset)))

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
        epoch_start_time = time.time()  # Record the start time of the epoch

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
            
            batch_number = total/32
            if batch_number % 100 ==0:
              print("This is batch {} of epoch {}".format(batch_number,epoch+1))

        training_loss /= len(train_loader)
        training_accuracy = correct / total

        model.eval()

        # Save the model after training 1 epoch
        if save_model_flag:
            torch.save(model.state_dict(), "pytorch_model_{}.pth".format(epoch+1))

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
        epoch_end_time = time.time()  # Record the end time of the epoch
        epoch_time = epoch_end_time - epoch_start_time  # Calculate the time taken for the epoch
        epoch_time_str = "{:0>2}:{:05.2f}".format(int(epoch_time // 60), epoch_time % 60)

        file_path = 'output.txt'
        with open(file_path, 'a') as file:
            # Assuming you have the variables epoch, epochs, training_loss, training_accuracy, val_loss, and accuracy defined
            content = f"Epoch [{epoch + 1}/{epochs}] ({epoch_time_str}))- Training Loss: {training_loss:.4f} - Training Accuracy: {100 * training_accuracy:.2f}% - Validation Loss: {val_loss:.4f} - Validation Accuracy: {100 * accuracy:.2f}%"

            # Print the content to the console
            print(content)

            # Write the content to the file
            print(content, file=file)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define number of epochs
epochs = 50

# Move the model to the device
model = nn.DataParallel(model)
model = model.to(device)

# Train the model
train_pytorch_model(model, training_loader, validation_loader, criterion, optimizer, epochs, device, save_model_flag=True)