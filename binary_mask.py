# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 19:33:26 2024

@author: krish
"""

import os
import cv2


# Specify the input and output directories
path = 'D:/Semantic segmentation project/cityscapes_data/Train_data/'
output_folder = 'D:/Semantic segmentation project/cityscapes_data/Train_mask/'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over each file in the input folder
P = sorted(os.listdir(path))
for i in P:
    image = cv2.imread(path+i,cv2.IMREAD_COLOR)
    

    # Threshold the image to create a binary mask
    ret, mask = cv2.threshold(image, 138, 255, cv2.THRESH_BINARY)

    cv2.imwrite(output_folder+i+"_label.jpg",mask)
    
