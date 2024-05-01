# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 23:03:14 2024

@author: krish
"""

import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
import torch.nn as nn
import numpy as np

model = torch.load('model_1.pth')

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Downsample path
        self.down1 = self.contracting_block(in_channels, 64)
        self.down2 = self.contracting_block(64, 128)
        self.down3 = self.contracting_block(128, 256)
        self.down4 = self.contracting_block(256, 512)
 
        # Upsample path
        self.up1 = self.expanding_block(512, 256)
        self.up2 = self.expanding_block(512, 128)
        self.up3 = self.expanding_block(256, 64)
        self.up4 = self.expanding_block(128, 64)
        # Output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block

    def expanding_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )
        return block

    def forward(self, x):
        # Downsample path
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        # Upsample path
        x = self.up1(x4)
        #print("After UP1, the size of X is: ",x.shape)
        x = torch.cat([x, x3], dim=1)
        #print("Post concatenation with x3, the size of x is: ",x.shape)
        x = self.up2(x)
        #print("After UP2, the size of x is: ",x.shape)
        x = torch.cat([x, x2], dim=1)
        #print("Post concatenation with x2, the size of x is: ",x.shape)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up4(x)
        # Output
        x = self.out(x)
        return x
# Rest of the code remains the same
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class TestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

new_path = "D:/Semantic segmentation project/cityscapes_data/predicted/"
if not os.path.isdir(new_path):
    os.mkdir(new_path)
def test_model(model, test_dataloader, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    transform = transforms.ToTensor()

    with torch.no_grad():
        image_index = 0
        for images in test_dataloader:
            images = images.to(device)  
 
            outputs = model(images)
            predictions = torch.sigmoid(outputs)  
            
            for i in range(predictions.shape[0]):
                segmented_image = predictions[i].squeeze().cpu().numpy()  # Convert tensor to numpy ndarray
                segmented_image = (segmented_image * 255).astype(np.uint8)  # Scale values to 0-255 range
                segmented_image = (segmented_image > (threshold * 255)).astype(np.uint8) * 255  # Threshold the image
                segmented_image_pil = Image.fromarray(segmented_image)  # Create PIL Image from ndarray

                # Save the segmented image using PIL
                image_path = test_dataset.image_paths[image_index]
                image_name = os.path.basename(image_path)
                predicted_image_path = os.path.join(new_path, image_name)
                segmented_image_pil.save(predicted_image_path)  # Save the segmented image

                image_index += 1
                
data_dir = "D:/Semantic segmentation project/cityscapes_data/training_data/"
# Specify the directory path containing test images
test_image_dir = os.path.join(data_dir, "Train_data") 
test_image_files = os.listdir(test_image_dir)
test_image_paths = [os.path.join(test_image_dir, file) for file in test_image_files]

# Create a test dataset and data loader
test_dataset = TestDataset(test_image_paths,transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the trained model
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load('model_1.pth'))

# Test the model and get segmented images
test_model(model, test_dataloader)