import os

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

    # Loop through each image file in the folder
    for idx, image_file in enumerate(image_file_names, start=1):
        # Get the old full path of the image file
        old_image_path = os.path.join(file_path, image_file)

        # Extract the folder name and use it as the new file name
        folder_name = file_name
        new_file_name = f"{folder_name}_{idx}.png"  # Assuming you want to rename as PNG files
        new_image_path = os.path.join(file_path, new_file_name)

        # Rename the image file
        os.rename(old_image_path, new_image_path)
