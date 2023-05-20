import torch 
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from dataloader import loadtotensor


# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d (Conv2D)              (None, 45, 45, 32)        832       
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 22, 22, 32)        0         
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 18, 18, 48)        38448     
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 9, 9, 48)          0         
# _________________________________________________________________
# flatten (Flatten)            (None, 3888)              0         
# _________________________________________________________________
# dense (Dense)                (None, 256)               995584    
# _________________________________________________________________
# dense_1 (Dense)              (None, 84)                21588     
# _________________________________________________________________
# dense_2 (Dense)              (None, 7)                 595       
# =================================================================

# Define the model

import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Change input channels from 1 to 3
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3872, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128,80)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

model = ConvNet()

FOLDER_NAME = "data_simple"

train_loader, test_loader= loadtotensor("data/{}/".format(FOLDER_NAME))

model = ConvNet()

classes = [
    'beta', 'pm', 'Delta', 'gamma', 'infty', 'rightarrow', 'div', 'gt',
    'forward_slash', 'leq', 'mu', 'exists', 'in', 'times', 'sin', 'R', 
    'u', '9', '0', '{', '7', 'i', 'N', 'G', '+', '6', 'z', '}', '1', '8',
    'T', 'S', 'cos', 'A', '-', 'f', 'o', 'H', 'sigma', 'sqrt', 'pi',
    'int', 'sum', 'lim', 'lambda', 'neq', 'log', 'forall', 'lt', 'theta',
    'M', '!', 'alpha', 'j', 'C', ']', '(', 'd', 'v', 'prime', 'q', '=',
    '4', 'X', 'phi', '3', 'tan', 'e', ')', '[', 'b', 'k', 'l', 'geq',
    '2', 'y', '5', 'p', 'w'
]

# loss function = CrossEntropyLoss
# optimizer = Adam and learning rate = 0.001 (maybe 0.01 is better)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# number of epochs = 10
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backpropagation and optimization to the initial input
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # get the index with the highest probability
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        # a tensort made of 1 as True and 0 and False to compare with the true labels and predicted labes
        correctPreds = (predicted == labels)
        correct += correctPreds.sum().item()

    # Calculate accuracy and average loss for the epoch
    accuracy = 100 * correct / total
    average_loss = total_loss / len(train_loader)

    # Print epoch statistics
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {average_loss:.4f} - Accuracy: {accuracy:.2f}%")

# Evaluation on test set
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


#SAVE MODEL
torch.save(model.state_dict(), "model.pth")
