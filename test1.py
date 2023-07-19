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


# Load the pretrained model
model.load_state_dict(torch.load("model_saved.pth"))
model.eval()

# Define the class labels
# classes = [
#     'beta', 'pm', 'Delta', 'gamma', 'infty', 'rightarrow', 'div', 'gt',
#     'forward_slash', 'leq', 'mu', 'exists', 'in', 'times', 'sin', 'R', 
#     'u', '9', '0', '{', '7', 'i', 'N', 'G', '+', '6', 'z', '}', '1', '8',
#     'T', 'S', 'cos', 'A', '-', 'f', 'o', 'H', 'sigma', 'sqrt', 'pi',
#     'int', 'sum', 'lim', 'lambda', 'neq', 'log', 'forall', 'lt', 'theta',
#     'M', '!', 'alpha', 'j', 'C', ']', '(', 'd', 'v', 'prime', 'q', '=',
#     '4', 'X', 'phi', '3', 'tan', 'e', ')', '[', 'b', 'k', 'l', 'geq',
#     '2', 'y', '5', 'p', 'w'
# ]

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Preprocess the images
transform = transforms.Compose([
    transforms.Resize((45, 45)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

image_paths = ["data/test_image/1_test.png", "data/test_image/2_test.png", "data/test_image/4_test.png", "data/test_image/6_test.png", "data/test_image/7_test.png"]
preprocessed_images = []
for image_path in image_paths:
    image = Image.open(image_path)
    preprocessed_image = transform(image)
    print(preprocessed_image.shape)
    preprocessed_images.append(preprocessed_image)

batch_images = torch.stack(preprocessed_images)

# Perform inference
with torch.no_grad():
    outputs = model(batch_images)
    probabilities = torch.softmax(outputs, dim=1)
    # print("PROBABILITIES: ",probabilities)
    _, predicted_labels = torch.max(probabilities, dim=1)
    print("PREDICTED LABELS: ",predicted_labels)

predicted_class_names = [classes[label] for label in predicted_labels]

# Print the predicted class names for the three images
for i, image_path in enumerate(image_paths):
    print(f"Image: {image_path} - Predicted Class: {predicted_class_names[i]}")
