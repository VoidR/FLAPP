import torch.nn as nn
import torch.nn.functional as F

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(dim_in, dim_out)
 
    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
 

