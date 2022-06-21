from cv2 import magnitude
import torch
import torchvision
import torchsummary

from torch.utils.data import DataLoader
from dataset import CustomDataset
from train.model.resnet import ResNet50
from train.model.simple_cnn import CNN
from train.manager import Manager

CFG = {
    "train_image_path" : "../train_img",
    "test_image_path" : "../test_img",

    "train_batch_size" : 256,
    "test_batch_size" : 1,
    
    "learning_rate" : 1e-4,
    
    "epoch" : 100   
}

if __name__ == "__main__":
    # Device Initialize
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Dataset Initialize
    dataloader = {}
    train_dataset = CustomDataset(CFG["train_image_path"], "train")
    dataloader["train"] = DataLoader(train_dataset, batch_size = CFG["train_batch_size"], shuffle=True, num_workers=0)
    
    valid_dataset = CustomDataset(CFG["train_image_path"], "valid")
    dataloader["valid"] = DataLoader(valid_dataset, batch_size = CFG["train_batch_size"], shuffle=False, num_workers=0)
    
    test_dataset = CustomDataset(CFG["test_image_path"], "test")
    dataloader["test"] = DataLoader(test_dataset, batch_size = CFG["test_batch_size"], shuffle=False, num_workers=0)

    # Model Import
    # model = ResNet50()
    model = CNN()
    
    torchsummary.summary(model, (3, 256, 256), batch_size = 16)

    manager = Manager(model, dataloader, device, CFG)
    manager.train()
    
                
