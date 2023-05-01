import torch 
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Subset

FOLDER_NAME = "data_simple"


classes = ['!' '(' ')' '+' ',' '-' '.DS_Store' '0' '1' '2' '3' '4' '5' '6' '7' '8'
 '9' '=' 'A' 'C' 'Delta' 'G' 'H' 'M' 'N' 'R' 'S' 'T' 'X' '[' ']' 'alpha'
 'ascii_124' 'b' 'beta' 'cos' 'd' 'div' 'e' 'exists' 'f' 'forall'
 'forward_slash' 'gamma' 'geq' 'gt' 'i' 'in' 'infty' 'int' 'j' 'k' 'l'
 'lambda' 'ldots' 'leq' 'lim' 'log' 'lt' 'mu' 'neq' 'o' 'p' 'phi' 'pi'
 'pm' 'prime' 'q' 'rightarrow' 'sigma' 'sin' 'sqrt' 'sum' 'tan' 'theta'
 'times' 'u' 'v' 'w' 'y' 'z' '{' '}']


def loadtotensor(dir):

    # print(dir)
    # Define the number of images per folder
    num_images_per_folder = 100

    # Create the dataset with all the images
    dataset = ImageFolder(root=dir, transform= transforms.ToTensor())

    # Create a list of indices for each folder to select only the desired number of images
    folder_indices = []
    for folder_name in os.listdir(dir):
        if folder_name != ".DS_Store":

            folder_path = os.path.join(dir, folder_name + "/")
            folder_images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
            if folder_name == "exists" or folder_name == "in" or folder_name == "forall":
                folder_indices.extend(random.sample(range(len(folder_images)), 20))
            else: 
                folder_indices.extend(random.sample(range(len(folder_images)), num_images_per_folder))
    
    #print(len(folder_indices)) # should be 82(classes) * num_images_per_folder


    # Create a subset of the dataset with only the desired images
    subset = Subset(dataset, folder_indices)

    #print(len(subset)) # should be 82 * num_images_per_folder as well

    # Create a DataLoader object with a batch size of 32
    batch_size = 32
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)


    return dataloader


data_loader = loadtotensor("data/{}/".format(FOLDER_NAME))

# Get a random batch
batch = next(iter(data_loader))

# Print the shape of the first tensor in the batch
# print(batch[0].shape)


batch = next(iter(data_loader))

# Get the images and labels from the batch
images, labels = batch

# Normalize
images = (images - images.min()) / (images.max() - images.min())

# Make a grid of the images and convert it to a numpy array
grid = make_grid(images, nrow=8, padding=2)
grid = grid.permute(1, 2, 0).numpy()

plt.figure(figsize=(10, 10))
plt.imshow(grid)
plt.show()

# Show the labels of the images
print(labels.numpy())




# Get a random image from the batch
# i = random.randint(0, len(batch[0]))
# print("i: ", i, len(batch[0]))
# image = batch[0][10].permute(1, 2, 0)
# # #Display the image using matplotlib
# plt.imshow(image)
# plt.show()
