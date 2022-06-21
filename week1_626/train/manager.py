import torch
import torch.nn as nn
import numpy as np
import time
import sys

from sklearn import metrics
from class_number import CLASS_NUMBER

class Manager():
    def __init__(self, model, dataloader, device, config):
        self.model = model
        self.config = config
        
        self.train_loader = dataloader["train"]
        self.valid_loader = dataloader["valid"]
        self.test_loader = dataloader["test"]
        self.device = device
        
        self.init_optimizer()
        self.init_loss()

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam([
            {'params' : self.model.parameters()}],
            lr = self.config["learning_rate"], betas=(0.99, 0.999))
    
    def init_loss(self):
        self.criterion = nn.CrossEntropyLoss().to(self.device)
    
    def train(self):
        self.model.to(self.device)
        best_mae = 9999
        
        epochs = self.config["epoch"]        
        
        for epoch in range(1,epochs+1):
            self.train_loop(epoch)
            
            
    
    def train_loop(self, epoch):
        self.model.train()
        
        total_train_loss = []
        tmp_train_loss = []
        
        step = 1
        steps = len(self.train_loader)
        prev_time = time.time()
        
        total_output = np.empty((0,10))
        total_label = np.empty((0,10))
        
        for img, label in iter(self.train_loader):
            img, label = img.float().to(self.device), label.float().to(self.device)
            self.optimizer.zero_grad()
            
            model_output = self.model(img)
            
            loss = self.criterion(model_output.squeeze(1), label)
            loss.backward()
            self.optimizer.step()
                
            tmp_train_loss.append(loss.item())
            total_train_loss.append(loss.item())
            
            total_output = np.append(total_output, model_output.detach().numpy(), axis = 0)
            total_label = np.append(total_label, label.detach().numpy(), axis = 0)
            
            if step % 2 == 0:
                print("[{}] STEP : ({}/{}) | TRAIN LOSS : {} | Time : {}]".format(
                epoch, step, steps, np.mean(tmp_train_loss), time.time()-prev_time))
                sys.stdout.flush()
                prev_time = time.time()
                self.calculation(total_output, total_label)
                
            step += 1
        
    def calculation(self, output_list, label_list):        
        output_list = np.argmax(output_list, axis = 1)
        label_list = np.argmax(label_list, axis = 1)
        
        class_list = []
        for class_name in CLASS_NUMBER:
            class_list.append(class_name)
        
        metric = metrics.classification_report(label_list, output_list, digits=10, target_names=class_list)
        
        print(metric)
            