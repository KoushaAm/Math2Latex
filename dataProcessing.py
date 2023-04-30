import torch 
import matplotlib.pyplot as plt
import os
import random
from torchvision import transforms
from PIL import Image


#get a random image from the dataset

DATA_SIMPLE_DIR = "data/data_simple/"


# Data visualization tool
# gets the directory(as str) and the type of operator or operand (as str)
def get_random_image(dir, type):
    dir += type
    random_image_name = random.choice(os.listdir(dir))
    #open the image file
    random_image_path = os.path.join(dir,  random_image_name)
    img = Image.open(random_image_path)
    #convert to tensor
    transform = transforms.ToTensor()
    tensor_image = transform(img)
    #convert to PIL image
    pil_image = transforms.ToPILImage()(tensor_image)
    #plot the image
    plt.imshow(pil_image)
    plt.show()
    return pil_image


img = get_random_image(DATA_SIMPLE_DIR, "infty")