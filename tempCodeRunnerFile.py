import os
import random

# Set the directory path
directory = 'data/data_simple'

# Get a list of file names in the directory
file_names = os.listdir(directory)

# Loop through each file in the directory
for file_name in file_names:
    if file_name == ".DS_Store":
        continue
    # Get the full path of the file
    file_path = os.path.join(directory, file_name)
    
    # Get the list of image file names in the directory
    image_file_names = [name for name in os.listdir(file_path) if name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
    # If there are already 300 or fewer images in the file, skip it
    if len(image_file_names) <= 300:
        continue
    
    # Determine how many images to delete
    num_to_delete = len(image_file_names) - 300
    
    # Randomly select and delete images until there are only 300 remaining
    for i in range(num_to_delete):
        # Choose a random image file to delete
        file_to_delete = os.path.join(file_path, random.choice(image_file_names))
        
        # Delete the file
        os.remove(file_to_delete)
        
        # Remove the file name from the list of image file names
        image_file_names.remove(os.path.basename(file_to_delete))