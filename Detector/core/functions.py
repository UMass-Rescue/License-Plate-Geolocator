import os
import cv2
import random
import numpy as np
import tensorflow as tf
from core.utils import read_class_names
from core.config import cfg

def crop_objects(img, data, path, allowed_classes, img_or_vid):
    boxes, scores, classes, num_objects = data
    #create dictionary to hold count of objects for image name
    counts = dict()
    class_name = 'license_plate'
    if class_name in allowed_classes:
        counts[class_name] = counts.get(class_name, 0) + 1
        # get box coords
        xmin, ymin, xmax, ymax = boxes[0][0]
        # crop detection from image (take an additional 5 pixels around all edges)
        img_height = img.shape[1]
        img_width = img.shape[0]
        if img_or_vid == 'image':
            cropped_img = img[int(xmin*img_width):int(xmax*img_width), int(ymin*img_height):int(ymax*img_height)]
        elif img_or_vid == 'video':
            cropped_img = img[int(xmin):int(xmax), int(ymin):int(ymax)]
        # construct image name and join it to path for saving crop properly
        img_name = class_name + '_detection_' + '.png'
        img_path = os.path.join(path, img_name )
        # save image
        if cropped_img.shape[0] !=0: # Avoids saving if no detections were made
            cv2.imwrite(img_path, cropped_img)