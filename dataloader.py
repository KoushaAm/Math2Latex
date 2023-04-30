import torch 
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

def loadtotensor(dir):
    # Define the number of images per folder
    num_images_per_folder = 10

    # Create the dataset with all the images
    dataset = ImageFolder(root=dir)

    # Create a list of indices for each folder to select only the desired number of images
    folder_indices = []
    for folder_name in os.listdir(dir):
        if folder_name != ".DS_Store":
            folder_path = os.path.join(dir, folder_name + "/")
            folder_images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
            if folder_images:
                folder_indices.extend([dataset.samples.index((os.path.join(folder_path, image_name), dataset.class_to_idx[folder_name])) for image_name in folder_images[:num_images_per_folder]])

    # Create a subset of the dataset with only the desired images
    transform = transforms.Compose([transforms.ToTensor()])
    subset = Subset(dataset, folder_indices)
    for i in subset:
        subset.append(transform.totensor(i))
    # Create a DataLoader object with a batch size of 32
    batch_size = 32
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    return dataloader


data_loader = loadtotensor("data/data_simple/")

# Get a random batch
batch = next(iter(data_loader))

# Get a random image from the batch
#for i in batch:
    #image = batch[i]
    #transform = transforms.ToTensor()
    #tensor_image = transform(image)
        #convert to PIL image
    #pil_image = transforms.ToPILImage()(tensor_image)
    # Display the image using matplotlib
    #plt.imshow(image)
    #plt.show()
