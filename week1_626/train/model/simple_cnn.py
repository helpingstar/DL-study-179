from sklearn.decomposition import KernelPCA
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 3x256x256 ->  32x128x128
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        # 32x128x128 -> 32x64x64
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)        
        
        # 32x64x64 -> 16x32x32
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        
        # 16x32x32-> 16x16x16
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 16x16x16 -> 8x8x8
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU()
        )

        self.tail_1 = nn.Linear(512, 32)
        
        self.tail_2 = nn.Sequential(
                nn.BatchNorm1d(32),
                nn.Dropout(0.2),
                nn.LeakyReLU(),
                nn.Linear(32, 10)
        )

        self.output = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool_1(x)
        
        x = self.conv2(x)
        x = self.maxpool_2(x)
        
        x = self.conv3(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.tail_1(x)
        x = self.tail_2(x)
        
        x = self.output(x)
                
        return x
