from inspect import ArgSpec
import torch
import torchsummary
import neptune.new as neptune
import argparse

from torch.utils.data import DataLoader
from dataset import CustomDataset
from train.model.resnet import ResNet50
from train.model.seresnet import SEResNet50
from train.model.efficient_net import EfficientNet_b0
from train.model.simple_cnn import CNN
from train.manager import Manager
from train.dataset import DataManger

CFG = {
    "save_path" : "./saved",
    "model_name" : "seresnet",
    
    "train_image_path" : "../train_img",
    "test_image_path" : "../test_img",

    "train_batch_size" : 128,
    "test_batch_size" : 4,
    
    "learning_rate" : 1e-4,
    
    "epoch" : 300   
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--model_name", default="sample")
    parser.add_argument("--enable_log", action='store_true')
    parser.add_argument("--enable_multigpu", action = 'store_true')
    args = parser.parse_args()
    config = CFG
    
    if args.enable_multigpu:
        config["multi_gpu"] = True
    else:
        config["multi_gpu"] = False        
    
    config["model_name"] = args.model_name
    
    neptune_instance = None
    if args.enable_log:
        neptune_instance = neptune.init(
            project="hsh-dev/dl-study",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZDNlMTkxZS0wOTE4LTRhYzUtODUzNS1hNGUyOTkzMTU0MjgifQ==",
            
        )
        neptune_instance["parameters"] = config
    
    # Device Initialize
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    
    # Dataset Initialize
    datamanager = DataManger(config)
    # datamanager.small_class_init()
    
    # Model Import
    # model = ResNet50()
    # model = SEResNet50()
    # model = CNN()
    model = EfficientNet_b0()
    
    model.to(device)
    
    torchsummary.summary(model, (3, 256, 256), batch_size = 16)

    manager = Manager(model, datamanager, device, config, neptune_instance)
    manager.train()
    
                
