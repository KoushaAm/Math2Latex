import torch 
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset



def loadtotensor(dir, type, number):

    # Set the path to your dataset folder
    data_path = "data/data_simple/"

    # Define the number of images per folder
    num_images_per_folder = 10

    # Create the dataset with all the images
    dataset = ImageFolder(root=data_path)

    # Create a list of indices for each folder to select only the desired number of images
    folder_indices = []
    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        folder_images = os.listdir(folder_path)
        folder_indices.extend([dataset.index((os.path.join(folder_path, image_name))) for image_name in folder_images[:num_images_per_folder]])

    # Create a subset of the dataset with only the desired images
    subset = Subset(dataset, folder_indices)

    # Create a DataLoader object with a batch size of 32
    batch_size = 32
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    print (len(dataloader))


    # Iterate over the data in batches
    #for data in dataloader:
        
    # Do something with the data