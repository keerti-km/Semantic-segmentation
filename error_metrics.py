# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 23:28:29 2024

@author: krish
"""

import os
import cv2
import numpy as np

groundtruth_path = "D:/Semantic segmentation project/cityscapes_data/train_mask/"
pred_path="D:/Semantic segmentation project/cityscapes_data/predicted/"
P = sorted(os.listdir(groundtruth_path))


gt_files = os.listdir(groundtruth_path)

def calculate_MAE(ground_truth_mask,predicted ):
    absolute_errors = np.abs(ground_truth_mask - predicted)
    mae = np.mean(absolute_errors)
    return mae

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_average_precision(mask1, mask2):
    num_thresholds = 11  # Number of IOU thresholds to evaluate
    thresholds = np.linspace(0.5, 1.0, num_thresholds)
    average_precision = 0.0

    for threshold in thresholds:
        tp, fp, fn = 0, 0, 0

        for i in range(len(mask1)):
            iou = calculate_iou(mask1[i], mask2[i])

            if iou >= threshold:
                tp += 1
            else:
                fp += 1

        fn = len(mask1) - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        average_precision += precision

    average_precision /= num_thresholds
    return average_precision

total_mae=0
total_map=0
mae_errors = []
dice_scores = []
# Iterate through the files and access the corresponding file in the other folder
for i, gt_file in enumerate(gt_files):
    # Load ground truth image
    gt_image = cv2.imread(os.path.join(groundtruth_path, gt_file))
    
    # Construct the file name for the corresponding predicted image
    pred_image = os.listdir(pred_path)[i]
    
    # Load predicted image
    pred_image = cv2.imread(os.path.join(pred_path, pred_image))
    mae = calculate_MAE(gt_image, pred_image)
    map_score = calculate_average_precision(gt_image, pred_image)
    
    total_mae+=mae
    total_map+=map_score
    abs_diff = np.abs(gt_image - pred_image)
    
    # Calculate mean absolute error for the image
    mae = np.mean(abs_diff)
    mae_errors.append(mae)
    
    # Calculate dice coefficient
    intersection = np.logical_and(gt_image, pred_image)
    dice_score = 2.0 * np.sum(intersection) / (np.sum(gt_image) + np.sum(pred_image))
    dice_scores.append(dice_score)

# Calculate the overall mean absolute error
overall_mae = np.mean(mae_errors)

# Calculate the overall dice coefficient
overall_dice = np.mean(dice_scores)
dice_percentage = overall_dice * 100


# Print the results

print("Mean Absolute Error (MAE):", overall_mae)
print("Dice Coefficient:", overall_dice)
print("Dice Coefficient Percentage:", dice_percentage)
print("mean mae: ",total_mae/2000)