from PIL import Image
import os
from loguru import logger

def reduce_resolution(image_path, reduction_percentage):
    # Open an image file
    #logger.debug(f"{image_path}")
    with Image.open(image_path) as img:
        # Calculate the new dimensions
        new_width = int(img.width * (1 - reduction_percentage / 100))
        new_height = int(img.height * (1 - reduction_percentage / 100))

        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)
        # Save the resized image
        resized_img.save(f"{image_path}_{reduction_percentage}pc.jpg")



def list_subfolders_files(directory):
    #logger.debug("list_subfolders_files()")
    #logger.debug(f"{directory}")
    # Iterate over the directories and files in the given directory
    for root, dirs, files in os.walk(directory):
        #logger.info("in first for")
        for subdir in dirs:
            #logger.info("in second for")
            #print(f"Subfolder: {subdir}")
            # Construct the path to the subdirectory
            subdir_path = os.path.join(root, subdir)
            # List files in the subdirectory
            subdir_files = os.listdir(subdir_path)
            for file in subdir_files:
                #logger.info("in third for")
                #print(f"    File: {file}")
                if not file.startswith('.') and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    try:
                        # Reduce the resolution by 45%
                        reduce_resolution(f"{directory}/{subdir}/{file}", 90)
                        # Reduce the resolution by 85%
                        reduce_resolution(f"{directory}/{subdir}/{file}", 93)
                        # Reduce the resolution by 97%
                        reduce_resolution(f"{directory}/{subdir}/{file}", 97)
                    except IOError:
                        print(f"Cannot open {file}.")
                else:
                    print(f"Skipping {file} as it is not an image.")

        break

logger.info("Program start")
# Replace 'directory_path' with the path to the directory you want to run it on
# folder structure must represent images folder https://drive.google.com/drive/folders/1RSvWruc5AvOmGoB3RRCimcfynTBXTYxU
list_subfolders_files('D:\geolocator_temp\images_with_reduced_res')
logger.info("Program end")