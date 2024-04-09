from torch import nn
from torch.nn.functional import max_pool2d, relu
 
 
class LeNet(nn.Module):
    def __init__(self, dim_in=1, dim_out=10):
        super(LeNet, self).__init__()
        self.Conv1 = nn.Conv2d(dim_in, 6, 5)
        self.Conv2 = nn.Conv2d(6, 16, 5)
        self.Conv3 = nn.Linear(16 * 4 * 4, 120)
        self.fc1 = nn.Linear(120, 64)
        self.fc2 = nn.Linear(64, dim_out)
 
    def forward(self, x):
        x = max_pool2d(relu(self.Conv1(x)), kernel_size=2)
        x = max_pool2d(relu(self.Conv2(x)), kernel_size=2)
        x = x.view(-1, 16 * 4 * 4)
        x = relu(self.Conv3(x))
        x = relu(self.fc1(x))
        x = self.fc2(x)
 
        return x
