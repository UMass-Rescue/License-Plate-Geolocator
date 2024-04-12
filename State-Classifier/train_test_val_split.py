# This script is used to split the image folders into train, test, and validation sets as the format needed by fastai
import splitfolders

# splitfolders.ratio("D:\geolocator_temp\images", # The location of dataset
#                    output="D:\geolocator_temp\images_split", # The output location
#                    seed=42, # The number of seed
#                    ratio=(.7, .15, .15), # The ratio of splited dataset
#                    group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
#                    move=False # If you choose to move, turn this into True
#                    )

splitfolders.ratio("D:\geolocator_temp\images_sample", # The location of dataset
                   output="D:\geolocator_temp\images_sample_split", # The output location
                   seed=42, # The number of seed
                   ratio=(.5, .25, .25), # The ratio of splited dataset
                   group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
                   move=False # If you choose to move, turn this into True
                   )