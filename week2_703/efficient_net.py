import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class MBConv(nn.Module):
    def __init__(self, in_channel, out_channel, layers, kernel, exp_factor, downscaling = False) -> None:
        super().__init__()
        self.exp_factor = exp_factor
                
        self.in_c = in_channel
        self.exp_c = in_channel * self.exp_factor
        self.out_c = out_channel
        
        self.layers = layers
        
        self.kernel = 3
        self.stride = 1
        if kernel == 5:
            self.kernel = 5
            self.stride = 2
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.in_c, self.exp_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.exp_c),
            nn.ReLU6(),
            nn.Conv2d(self.exp_c, self.exp_c, kernel_size=self.kernel, stride=self.stride, padding=1, groups=self.exp_c),
            nn.BatchNorm2d(self.exp_c),
            nn.ReLU6(),
            nn.Conv2d(self.exp_c, self.out_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_c)
        )
        
        
    def forward(self, x):
        shortcut = x
        x = self.bottleneck(x)
        output = x + shortcut
        
        return output



class EfficientNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO : Implement Network
        
        
        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )
            
        self.MBConv1_3 = MBConv(in_channel = 32, out_channel = 16, 
                                layers = 1, kernel = 3, exp_factor=1)
        self.MBConv6_3 = MBConv(in_channel = 16, out_channel = 24, 
                                layers = 2, kernel = 3, exp_factor=6)
        self.MBConv6_5 = MBConv(in_channel = 24, out_channel = 40,
                                layers = 2, kernel = 5, exp_factor=6, downscaling = True)
        
        
        
    def forward(self, x):
        # TODO : Implement Network
        ## Sample Convolution, delete this
        x = self.head(x)
        x = self.mbc_1(x)
        return x
    




if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    
    batch_size = 4
    input = torch.randn((batch_size, 24, 54, 54))
    model = EfficientNet()
    output = model(input)
    
    print(output.shape)
    
    model.to(device)
    summary(model, (24, 54, 54))
        
