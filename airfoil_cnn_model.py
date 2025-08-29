import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import torch.cuda.amp as amp
import pandas as pd
from PIL import Image
import os

class AirfoilCNN(nn.Module):
    def __init__(self, input_height, input_width):
        super().__init__()
        # Image branch
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Enhancement: Stabilize training
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        self.conv_output_size = self._get_conv_output_size(input_height, input_width)

        # AoA branch
        self.aoa_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )
        
        # Combined head
        self.head = nn.Sequential(
            nn.Linear(self.conv_output_size + 16, 512),
            nn.Dropout(0.2),  # Enhancement: Regularization
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def _get_conv_output_size(self, h, w):
        with torch.no_grad():
            dummy_input = torch.zeros(1,1,h,w)
            output = self.conv_layers(dummy_input)
            print (f"conv output size: {output.shape[1]}")
            return output.shape[1]

    def forward(self, x_img, x_aoa):
        img_features = self.conv_layers(x_img)
        aoa_features = self.aoa_fc(x_aoa)
        combined = torch.cat([img_features, aoa_features], dim=1)
        return self.head(combined)
