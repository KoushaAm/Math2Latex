import torch 
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from sklearn.preprocessing import LabelEncoder



FOLDER_NAME = "data_simple"


classes = ['-','!', '(', ')','[', ']', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8',
 '9', '=', 'A', 'C', 'Delta', 'G', 'H', 'M', 'N', 'R', 'S', 'T', 'X', 'alpha',
 '|', 'b', 'beta', 'cos', 'd', 'div', 'e', 'exists', 'f', 'forall',
 'forward_slash', 'gamma', 'geq', 'gt', 'i', 'in', 'infty', 'int', 'j', 'k', 'l',
 'lambda', 'ldots', 'leq', 'lim', 'log', 'lt', 'mu', 'neq', 'o', 'p', 'phi', 'pi',
 'pm', "prime" , 'q', 'rightarrow', 'sigma', 'sin', 'sqrt', 'sum', 'tan', 'theta',
 'times', 'u', 'v', 'w', 'y', 'z', '{', '}']


# initialize the label encoder
label_encoder = LabelEncoder().fit(classes)
# fit the label encoder to the list of classes

# encoded labels
encoded_classes = label_encoder.transform(classes)
#print(encoded_labels) #why does it print [0] only?



def loadtotensor(dir):

    # print(dir)
    # Define the number of images per folder
    num_images_per_folder = 100

    # Create the dataset with all the images
    dataset = ImageFolder(root=dir, transform= transforms.ToTensor())

    # Create a list of indices for each folder to select only the desired number of images
    folder_indices = []
    folder_labels = []
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    folder_images=[]
    for folder_name in os.listdir(dir):
        print(folder_name)
        
        
        if folder_name != ".DS_Store":

            folder_path = os.path.join(dir, folder_name + "/")
            folder_images.extend([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
            label = label_encoder.transform([folder_name])

            if folder_name in ["exists", "in", "forall"]:
                indices = random.sample(range(len(folder_images)), 20)
                folder_indices.extend(indices)
                
                
            else: 
                folder_indices.extend(random.sample(range(len(folder_images)), num_images_per_folder))
                
            

            # print(folder_name)
    
            label = label_encoder.transform([folder_name])
            folder_labels.append(label)

        # print(folder_indices)
        # print("new folder")
               

    # Shuffle the indices

    #print(len(folder_indices)) # should be 82(classes) * num_images_per_folder

    # Encode the label using LabelEncoder
    folder_labels = torch.tensor(folder_labels, dtype=torch.int64)
    folder_labels = folder_labels.tolist()
    folder_labels.extend([label] * len(folder_images))
    #print(folder_indices)
    # print(label_encoder.inverse_transform(label))
    random.shuffle(folder_indices)
    # Create a subset of the dataset with only the desired images
    #print(folder_indices)
    subset = Subset(dataset, folder_indices)

    #print(len(subset)) # should be 82 * num_images_per_folder as well

    # Create a DataLoader object with a batch size of 32
    batch_size = 32
    dataloader = DataLoader(subset, batch_size=batch_size, drop_last=True, shuffle=True)


    return dataloader




# Create a DataLoader object
data_loader = loadtotensor("data/{}/".format(FOLDER_NAME))


# # Create a figure with 5 rows and 2 columns to display 10 batches
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i in range(4):
    # Get a random batch
    batch = next(iter(data_loader))

    # Get the images and labels from the batch
    images, labels = batch
    print(labels)

    images = (images - images.min()) / (images.max() - images.min())
    #plt.figure(figsize=(10, 10))
    #plt.imshow(grid)
    #plt.show()

    # Make a grid of the images and convert it to a numpy array
    grid = make_grid(images, nrow=8, padding=2)
    grid = grid.permute(1, 2, 0).numpy()

    # Plot the grid in a subplot
    row = i // 2
    col = i % 2
    axs[row, col].imshow(grid)
    axs[row, col].set_title(f"Batch {i+1}")

# Show the plot
plt.tight_layout()
plt.savefig('batches.png')
plt.show()

for i in range(0, 20):
    batch = next(iter(data_loader))
    images, labels = batch
     # Select a random image from the batch
    idx = random.randint(0, len(images) - 1)
    image = images[idx].permute(1, 2, 0)
     # Display the image using matplotlib
    plt.imshow(image)
    plt.title(label_encoder.inverse_transform([labels[idx]])[0])
    plt.show()
