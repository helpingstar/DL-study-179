from inspect import ArgSpec
import torch
import torchsummary
import neptune.new as neptune
import argparse

from torch.utils.data import DataLoader
from dataset import CustomDataset
from train.model.resnet import ResNet50
from train.model.seresnet import SEResNet50
from train.model.simple_cnn import CNN
from train.manager import Manager
from model.loader import load_model


CFG = {
    "save_path" : "./saved",
    "model_name" : "seresnet",
    
    "train_image_path" : "../train_img",
    "test_image_path" : "../test_img",

    "train_batch_size" : 2,
    "test_batch_size" : 2,
    
    "learning_rate" : 1e-4,
    
    "epoch" : 100   
}

if __name__ == "__main__":
    # Device Initialize
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    
    # Dataset Initialize
    dataloader = {}
    train_dataset = CustomDataset(CFG["train_image_path"], "train")
    dataloader["train"] = DataLoader(train_dataset, batch_size = CFG["train_batch_size"], shuffle=True, num_workers=0, drop_last=True)
    
    valid_dataset = CustomDataset(CFG["train_image_path"], "valid")
    dataloader["valid"] = DataLoader(valid_dataset, batch_size = CFG["train_batch_size"], shuffle=False, num_workers=0, drop_last=True)
    
    test_dataset = CustomDataset(CFG["test_image_path"], "test")
    dataloader["test"] = DataLoader(test_dataset, batch_size = CFG["test_batch_size"], shuffle=False, num_workers=0,  drop_last=False)

    # Model Import
    # model = ResNet50()
    model = SEResNet50()
    # model = CNN()

    model = load_model(model, "./saved", "seresnet", device)
    
    ## MULTI GPU    
    # NGPU = torch.cuda.device_count()
    # if NGPU > 1:
    #     model = torch.nn.DataParallel(model, device_ids=list(range(NGPU)))
    #     torch.multiprocessing.set_start_method('spawn')

    
    model.to(device)
    
    torchsummary.summary(model, (3, 256, 256), batch_size = 16)

    manager = Manager(model, dataloader, device, CFG)
    manager.test()
    