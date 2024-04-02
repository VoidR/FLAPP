import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(dim_in*dim_in, 128)  # MNIST images are 28x28 pixels
        self.fc2 = nn.Linear(128, dim_out)     # 10 classes for MNIST digits

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
