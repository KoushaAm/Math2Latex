import torch 
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

#get a random image from the dataset
DATA_SIMPLE_DIR = "data/data_simple/"

#classes of the dataset

#get the categories of the dataset

def get_categories(dir):
    symbols = sorted([symbol for symbol in os.listdir(dir) if symbol.isdigit()])
    return symbols

symbles = get_categories(DATA_SIMPLE_DIR)
print(symbles, len(symbles)) # 83 classes




# Data visualization tool
# gets the directory(as str) and the type of operator or operand (as str)
def get_random_image(dir, type):
    dir += type
    random_image_name = random.choice(os.listdir(dir))

    #open the image 
    random_image_path = os.path.join(dir,  random_image_name)
    img = Image.open(random_image_path)


    #convert to tensor
    transform = transforms.ToTensor()
    tensor_image = transform(img)


    #convert to PIL image
    pil_image = transforms.ToPILImage()(tensor_image)
    #plot the image
    visual_image(pil_image)
    

def visual_image(img):
    plt.imshow(img)
    plt.show()
    return img

visual_image(get_random_image(DATA_SIMPLE_DIR, "9"))
