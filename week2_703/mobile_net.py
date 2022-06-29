import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class MobileNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO : Implement Network
        ## Sample Convolution, delete this
        self.conv = nn.Conv2d(3, 10, kernel_size = 3, stride = 2, padding = 1)
    
    def forward(self, x):
        # TODO : Implement Network
        ## Sample Convolution, delete this
        x = self.conv(x)
        return x
    




if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    
    batch_size = 4
    input = torch.randn((batch_size, 3, 224, 224))
    model = MobileNet()
    output = model(input)
    
    print(output.shape)
    
    model.to(device)
    summary(model, (3, 224, 224))
        
