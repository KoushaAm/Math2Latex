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


