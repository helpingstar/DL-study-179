import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from model.submodule.swish import Swish

class MBConv(nn.Module):
    def __init__(self, in_c, out_c, kernel, exp_f = 6, downscaling = False) -> None:
        super().__init__()
        
        initial_stride = 1
        if downscaling:
            initial_stride = 2
        
        same_kernel = 3
        same_padding = 1
        if kernel == 5:
            same_kernel = 5
            same_padding = 2
        
        self.mbconv = nn.Sequential(
            nn.Conv2d(in_c, in_c * exp_f, kernel_size=1, stride = initial_stride, padding = 0),
            nn.BatchNorm2d(in_c * exp_f),
            nn.ReLU6(),
            nn.Conv2d(in_c * exp_f, in_c * exp_f, kernel_size = same_kernel, stride = 1, padding = same_padding, groups = in_c * exp_f),
            nn.BatchNorm2d(in_c * exp_f),
            nn.ReLU6(),
            nn.Conv2d(in_c * exp_f, out_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_c)
            )
        
    def forward(self, x):
        
        return self.mbconv(x)


class ModuleBlock(nn.Module):
    def __init__(self, in_c, out_c, layers, kernel, exp_f, downscaling = False) -> None:
        super().__init__()
        self.layers = layers

        self.stage = nn.ModuleList([])
        
        if downscaling == False:
            self.stage.append(MBConv(in_c, out_c, kernel, exp_f, False))
        else:
            self.stage.append(MBConv(in_c, out_c, kernel, exp_f, True))
        
        for i in range(1, layers):
            self.stage.append(MBConv(out_c, out_c, kernel, exp_f, False))
        
        
    def forward(self, x):
        x = self.stage[0](x)
        
        for i in range(1, self.layers):
            shortcut = x
            x = self.stage[i](x)

            x = shortcut + x
        
        return x


class EfficientNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO : Implement Network
        
        channel_list = [32, 16, 24, 40, 80, 112, 192, 320]
        layer_list = [1, 2, 2, 3, 3, 4, 1]
        kernel_list = [3, 3, 5, 3, 5, 5, 3]
        expansion_list = [1, 6, 6, 6, 6, 6, 6]
        downscaling_list = [False, True, True, True, False, True, False]
        
        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )
        
        self.module_stage = nn.ModuleList([])
        
        for i in range(7):
            self.module_stage.append(
                    ModuleBlock(
                        channel_list[i], channel_list[i+1], 
                        layer_list[i], 
                        kernel_list[i], 
                        expansion_list[i],
                        downscaling_list[i]
                )
            )
            
        self.final_stage = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1280),
            Swish()
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, 10)
        self.softmax = nn.Softmax(dim = 1)
        
        
    def forward(self, x):
        # TODO : Implement Network
        x = self.head(x)
        for i in range(7):
            x = self.module_stage[i](x)

        x = self.final_stage(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        
        return x
    

def EfficientNet_b0():
    return EfficientNet()



if __name__ == "__main__":
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')    
    
    batch_size = 4
    input = torch.randn((batch_size, 3, 256, 256))
    model = EfficientNet()
    output = model(input)
    
    print(output.shape)
    
    model.to(device)
    summary(model, (3, 256, 256))
        
