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

data= loadtotensor("data/{}/".format(FOLDER_NAME))

for batch in data:
    output=model(batch[0])


torch.save(model, 'model.pth')

# Load the entire model
model = torch.load('model.pth')

model = torch.load('model.pth')

# Set the model in evaluation mode
model.eval()

# Perform inference on test data
label_mapping = ['beta', 'pm', 'Delta', 'gamma', 'infty', 'rightarrow', 'div', 'gt',
           'forward_slash', 'leq', 'mu', 'exists', 'in', 'times', 'sin', 'R', 
           'u', '9', '0', '{', '7', 'i', 'N', 'G', '+', '6', 'z', '}', '1', '8',
             'T', 'S', 'cos', 'A', '-', 'f', 'o', 'H', 'sigma', 'sqrt', 'pi',
               'int', 'sum', 'lim', 'lambda', 'neq', 'log', 'forall', 'lt', 'theta',
                 'M', '!', 'alpha', 'j', 'C', ']', '(', 'd', 'v', 'prime', 'q', '=',
                   '4', 'X', 'phi', '3', 'tan', 'e', ')', '[', 'b', 'k', 'l', 'geq',
                     '2', 'y', '5', 'p', 'w']

predictions=[]

with torch.no_grad():
    for batch in data:
        inputs = batch[0]
        labels = batch[1]
        outputs = model(inputs)
        _, predicted_indices = torch.max(outputs, 1)
        
        # Assign labels based on predicted indices
        predicted_labels = [label_mapping[idx] for idx in predicted_indices]
        predictions.extend(predicted_labels)

# Print the predictions
print(predictions)

