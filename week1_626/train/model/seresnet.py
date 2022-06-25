import torch
import torch.nn as nn
from train.model.resnet import ResNet

class SEBlock(nn.Module):
    def __init__(self, in_c, ratio = 16):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
        self.excitation = nn.Sequential(
            nn.Linear(in_c, in_c // ratio),
            nn.ReLU(),
            nn.Linear(in_c // ratio, in_c),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # CxHxW -> 1xHxW
        x = self.squeeze(x)
        # 1xHxW -> H*W
        x = x.view(x.size(0), -1)
        # H*W -> H*W / r -> H*W
        x = self.excitation(x)
        # H*W -> 1xHxW
        x = x.view(x.size(0), x.size(1), 1, 1)
        
        return x
        

class SEBottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_c, out_c, stride = 1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels = in_c, out_channels = out_c, kernel_size = 1, stride= 1, bias = False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(in_channels= out_c, out_channels = out_c, kernel_size=3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(in_channels= out_c, out_channels = out_c * SEBottleNeck.expansion, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_c * SEBottleNeck.expansion)
        )

        self.se_block = nn.Sequential(
            SEBlock(out_c * SEBottleNeck.expansion)
        )
        
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_c != SEBottleNeck.expansion * out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels = in_c, out_channels = out_c * SEBottleNeck.expansion, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_c * SEBottleNeck.expansion),
            )
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        res_out = self.residual_function(x)
        scale_out = self.se_block(res_out) * res_out
        
        x = scale_out + shortcut
        x = self.relu(x)
        
        return x
    
    
def SEResNet50():
    return ResNet(SEBottleNeck, [3, 4, 6, 3])


if __name__ == "__main__":
    x = torch.randn(3, 3, 256, 256)
    seresnet = SEResNet50()
    output = seresnet(x)
    print(output.size())
    
    