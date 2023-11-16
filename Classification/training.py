"""
Make all the imports:
"""

# imports for downloading files
import urllib.request
import zipfile
import os

#  imports for the network
import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""Download the Dataset for Classification:"""

# Get files in current working directory
files = os.listdir()

if 'Dataset' and 'Dataset.zip' not in files:
    # This url points to the download of the .zip file for the classification training
    url = 'https://www.dropbox.com/scl/fi/9sxb3s88hw2zr2f0bbvf9/Dataset.zip?rlkey=8s4bobjz0b7ee68vjt384cjk1&dl=1'

    # Download the zip file
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()

    # Specify the local filename for the downloaded zip file
    zip_filename = 'Dataset.zip'

    with open(zip_filename, 'wb') as f:
        f.write(data)

    # Unzip the downloaded file
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        # Extract all contents to the current working directory
        zip_ref.extractall()

"""Preprocessing and examination:"""

# Get the current folder paths of the dataset
Training_path = 'Dataset/Classification/Training'
Testing_path = 'Dataset/Classification/Test'

# Create the datasets
transform = transforms.Compose([transforms.ToTensor()])
training_dataset = torchvision.datasets.ImageFolder(Training_path, transform=transform)
testing_dataset = torchvision.datasets.ImageFolder(Testing_path)

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

map = {0: 'Fire', 1: 'No Fire'}

# Print one image
dataiter = iter(training_loader)
images, labels = next(dataiter)

for image, label in zip(images, labels):
    plt.figure()
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.title(map[label.item()])
    break

"""Define the Model:"""


class FLAME_SCOUT_Model(nn.Module):
    def __init__(self, num_classes):
        super(FLAME_SCOUT_Model, self).__init__()

        # Initial Convolutional Block
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        # Separable Convolution Blocks with Residual Connections
        self.conv_blocks = self._make_conv_blocks(8)

        # 1x1 Convolution for Residual Connection
        self.conv_residual = nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0)  # Adjusted stride

        # Final Convolutional Block
        self.final_conv = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn_final = nn.BatchNorm2d(8)

        # Global Average Pooling and Output Layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(8, num_classes)

    def _make_conv_blocks(self, size):
        return nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(size, size, kernel_size=3, padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(),
            nn.Conv2d(size, size, kernel_size=3, padding=1),
            nn.BatchNorm2d(size),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        previous_block_activation = x

        for block in self.conv_blocks:
            x = block(x)
            # Use 1x1 convolution for residual connection
            residual = self.conv_residual(previous_block_activation)
            residual = F.interpolate(residual, size=x.size()[2:],
                                     mode='nearest')  # Adjusted to match the spatial dimensions
            x = x + residual
            previous_block_activation = x

        x = F.relu(self.bn_final(self.final_conv(x)))

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


# Create an instance of the model
num_classes = 2
input_shape = (3, 254, 254)
model = FLAME_SCOUT_Model(num_classes)

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

