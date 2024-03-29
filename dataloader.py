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



FOLDER_NAME = "nums"


# classes = ['beta', 'pm', 'Delta', 'gamma', 'infty', 'rightarrow', 'div', 'gt',
#            'forward_slash', 'leq', 'mu', 'exists', 'in', 'times', 'sin', 'R', 
#            'u', '9', '0', '{', '7', 'i', 'N', 'G', '+', '6', 'z', '}', '1', '8',
#              'T', 'S', 'cos', 'A', '-', 'f', 'o', 'H', 'sigma', 'sqrt', 'pi',
#                'int', 'sum', 'lim', 'lambda', 'neq', 'log', 'forall', 'lt', 'theta',
#                  'M', '!', 'alpha', 'j', 'C', ']', '(', 'd', 'v', 'prime', 'q', '=',
#                    '4', 'X', 'phi', '3', 'tan', 'e', ')', '[', 'b', 'k', 'l', 'geq',
#                      '2', 'y', '5', 'p', 'w']

# get the list of numebrs only
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# initialize the label encoder
label_encoder = LabelEncoder().fit(classes)
# fit the label encoder to the list of classes

# encoded labels
encoded_classes = label_encoder.transform(classes)


def loadtotensor(dir, train_ratio=0.8):
    # Define the number of images per folder
    num_images_per_folder = 250

    # Create the dataset with all the images
    dataset = ImageFolder(root=dir, transform=transforms.ToTensor())

    # Create a list of indices for each folder to select only the desired number of images
    folder_indices = []
    folder_labels = []
    folder_images = []
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for folder_name in os.listdir(dir):
        if folder_name in classes:

            folder_path = os.path.join(dir, folder_name + "/")
            folder_images.extend([f for f in os.listdir(folder_path) if f.endswith('.png')])
            label = label_encoder.transform([folder_name])
            indices = random.sample(range(len(folder_images)), num_images_per_folder)
            # print(indices)
            

            folder_indices.extend(indices)
            folder_labels.extend([label] * len(indices))
  

    # Encode the label using LabelEncoder
    folder_labels = torch.tensor(folder_labels, dtype=torch.int64)
    folder_labels = folder_labels.tolist()
    folder_labels.extend([label] * len(folder_images))

    random.shuffle(folder_indices)

    # Calculate the number of training samples
    train_size = int(train_ratio * len(folder_indices))
    test_size = len(folder_indices) - train_size
    # Split the indices into training and test indices
    train_indices = folder_indices[:train_size]
    test_indices = folder_indices[train_size:]

    # Create subsets of the dataset for training and test data
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    # Create DataLoader objects for training and test data
    batch_size = 32
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader = DataLoader(test_subset, batch_size=batch_size, drop_last=True, shuffle=True)

    # Create a dictionary to track the labels that have been seen
    label_counts = {label: 0 for label in encoded_classes}

    for _, label in train_dataloader:
        # Update the label counts
        for l in label:
            l = l.item()
            label_counts[l] += 1

    # Check if any labels have not been seen at least once
    unseen_labels = [label for label, count in label_counts.items() if count < 50]

    if unseen_labels:
        # Resample the training data to include the unseen labels
        resampled_indices = []
        resampled_labels = []

    for idx, lbl in zip(train_indices, folder_labels):
        if lbl in unseen_labels:
            resampled_indices.append(idx)
            resampled_labels.append(lbl)
            label_counts[lbl[0]] += 1

            if all(count > 0 for count in label_counts.values()):
                break

    train_indices.extend(resampled_indices)
    folder_labels.extend(resampled_labels)

    # Shuffle the train indices and labels
    random.shuffle(train_indices)
    folder_labels = [folder_labels[idx] for idx in train_indices]

    # Update the training subset and dataloader
    train_subset = Subset(dataset, train_indices)
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, drop_last=True, shuffle=True)

    # print dictionary
    for label, count in label_counts.items():
        print(f"{label} or {classes[label]}: has {count} samples")

    return train_dataloader, test_dataloader



def show_batches(data_loader):

  # # # Create a figure with 5 rows and 2 columns to display 10 batches
  fig, axs = plt.subplots(2, 2, figsize=(10, 10))

  for i in range(4):
      # Get a random batch
      batch = next(iter(data_loader))

      images, labels = batch

      images = (images - images.min()) / (images.max() - images.min())


      grid = make_grid(images, nrow=8, padding=2)
      grid = grid.permute(1, 2, 0).numpy()

      # Plot the grid in a subplot
      row = i // 2
      col = i % 2
      axs[row, col].imshow(grid)
      axs[row, col].set_title(f"Batch {i+1}")

  # Show the plot
  plt.tight_layout()
  plt.show()


train_loader, test_loader = loadtotensor("data/{}/".format(FOLDER_NAME))
# show_batches(train_loader)


# for i in range(0, 20):
#     batch = next(iter(train_loader))
#     images, labels = batch
#      # Select a random image from the batch
#     idx = random.randint(0, len(images) - 1)
#     image = images[idx].permute(1, 2, 0)
#      # Display the image using matplotlib
#     plt.imshow(image)
#     plt.title(label_encoder.inverse_transform([labels[idx]])[0])
#     plt.show()


# # show_random_images(data_loader)
# print(len(classes))
