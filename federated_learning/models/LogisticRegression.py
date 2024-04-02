import torch.nn as nn
import torch.nn.functional as F

class LogisticRegressionModel(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
 
    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
 

