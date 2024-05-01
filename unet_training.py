# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 21:46:18 2024

@author: krish
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# Define the UNet model
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


# Define the training loop
def train_model(model, train_dataloader, optimizer, criterion, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        itr_num = 0  
        for images, masks in train_dataloader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            print("Epoch = ",epoch, " Iteration  = ",itr_num," Loss = ",loss.item()/32)
            itr_num+=1

        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.mask_paths is not None:
            mask_path = self.mask_paths[idx]
            mask = Image.open(mask_path).convert("L")  # Convert to grayscale mask

            if self.transform:
                mask = self.transform(mask)

            return image, mask

        return image
# Specify the directory path containing images and masks
data_dir = "D:/Semantic segmentation project/cityscapes_data/training_data/"

# Get a list of all image files in the directory
image_dir = os.path.join(data_dir, "Train_data")
image_files = os.listdir(image_dir)
image_paths = [os.path.join(image_dir, file) for file in image_files]

# Get a list of all mask files in the directory
mask_dir = os.path.join(data_dir, "train_mask")
mask_files = os.listdir(mask_dir)
mask_paths = [os.path.join(mask_dir, file) for file in mask_files]

# Verify that the number of images and masks match
assert len(image_paths) == len(mask_paths), "Number of images and masks should be the same"

# Rest of the code remains the same
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = CustomDataset(image_paths, mask_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

  

model = UNet(in_channels=3, out_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

num_epochs = 20
train_model(model, dataloader, optimizer, criterion, num_epochs)

torch.save(model.state_dict(),'model_1.pth')