
import os 
from PIL import Image

folder_dir_2014 = "data/2014"
folder_dir_2019 = "data/2019"
folder_dir_2016 = "data/2016"
folder_dir_train = "data/train"


#get the maximum dimenision of the images in the folder
#used for making a unified size for all images
def get_max_size(dir):

    max_width = 0
    max_height = 0
    file_name = ""


    for filename in os.listdir(dir):
        if filename.endswith(".bmp"):
            image_path = os.path.join(dir, filename)
            img = Image.open(image_path)
            width, height = img.size

            if ((width * height) > (max_width * max_height)):
                max_width = width
                max_height = height
                file_name = filename


    print(f"Max width in {dir}: {max_width}")
    print(f"Max height in {dir}: {max_height}")
    print(f"File name in {dir}: {file_name}")

            

get_max_size(folder_dir_2014)
get_max_size(folder_dir_2016)
get_max_size(folder_dir_2019)
get_max_size(folder_dir_train)
