import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, dim_in, num_classes, img_size):
        super(SimpleCNN, self).__init__()
        self.dim_in = dim_in
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(dim_in, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Adaptive pooling layer to ensure the output dimensions
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Input shape adaptation for grayscale images (e.g., MNIST)
        if x.shape[1] == 1:  # MNIST images are 1 channel
            x = x.repeat(1, self.dim_in, 1, 1)  # Repeat the grayscale image across channel dimension
        
        # Convolutional layers with ReLU and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        
        # Adaptive pooling to standardize output size
        x = self.pool(x)
        
        # Flattening the layer
        x = x.view(-1, 128 * 6 * 6)
        
        # Fully connected layers with ReLU and final output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
