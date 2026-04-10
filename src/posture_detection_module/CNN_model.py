import torch
import torch.nn as nn
import numpy as np

class SimpleIRA_CNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1) # output: (B, 32, 60, 80)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # output: (B, 64, 30, 40)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # output: (B, 128, 15, 20)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        
        # After conv + pooling: 128 * 7 * 10 = 8960
        self.fc1 = nn.Linear(128 * 7 * 10, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x comes in as (B, 60, 80)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        x = x.float()
        B, H, W = x.shape
        
        # Add channel dimension for Conv2d
        x = x.unsqueeze(1)                   # -> (B, 1, 60, 80)
        
        # Convolutional backbone
        x = self.relu(self.conv1(x))
        x = self.pool(x)                     # -> (B, 32, 30, 40)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)                     # -> (B, 64, 15, 20)
        
        x = self.relu(self.conv3(x))
        x = self.pool(x)                     # -> (B, 128, 7, 10)
        
        # Flatten
        x = x.view(B, -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x