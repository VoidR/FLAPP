import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, dim_in, num_classes, img_size):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(dim_in, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 计算经过特征提取层后的图像大小
        size_after_features = ((img_size - 11) // 4 + 1) // 2 // 2 // 2
        self.avgpool = nn.AdaptiveAvgPool2d((size_after_features, size_after_features))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * size_after_features * size_after_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x