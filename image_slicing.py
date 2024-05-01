# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 19:21:27 2024

@author: krish
"""

import cv2
import os

path = "D:/Semantic segmentation project/cityscapes_data/train/"

new_path = "D:/Semantic segmentation project/cityscapes_data/Train_data/"
new_path2="D:/Semantic segmentation project/cityscapes_data/Train_label/ "

if not os.path.isdir(new_path):
    os.mkdir(new_path)
if not os.path.isdir(new_path2):
    os.mkdir(new_path2)
    
P = sorted(os.listdir(path))
for i in P:
    image = cv2.imread(path+i)
    h, w, channels = image.shape
    half = w//2
    left_part = image[:, :half]
    right_part = image[:, half:] 
    
    # Perform graph cut segmentation
    #segmented_mask = perform_graph_cut(left_part)

    # Save the segmented mask
    cv2.imwrite(new_path+i+"_train.jpg",left_part)
    cv2.imwrite(new_path2+i+"_label.jpg",right_part)