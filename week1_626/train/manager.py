import torch
import torch.nn as nn
import numpy as np
import time
import sys, os

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from class_number import CLASS_NUMBER

class Manager():
    def __init__(self, model, dataloader, device, config, neptune_instance = None):
        self.model = model
        self.config = config
        
        self.train_loader = dataloader["train"]
        self.valid_loader = dataloader["valid"]
        self.test_loader = dataloader["test"]
        self.device = device
        
        self.multi_gpu = False
        
        self.enable_log = False
        if neptune_instance is not None:
            self.enable_log = True
            self.neptune = neptune_instance
        
        self.logs = {}
        
            
        self.init_optimizer()
        self.init_loss()
        
        if self.config["multi_gpu"]:
            self.init_multigpu()

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam([
            {'params' : self.model.parameters()}],
            lr = self.config["learning_rate"], betas=(0.99, 0.999))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch)
    
    def init_multigpu(self):
        NGPU = torch.cuda.device_count()
        if NGPU > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(NGPU)))
            torch.multiprocessing.set_start_method('spawn')
            self.multi_gpu = True
    
    
    def init_loss(self):
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    # def focal_loss(self):
    #     loss = nn.CrossEntropyLoss()
    
    def train(self):
        self.model.to(self.device)
        min_loss = 9999
        
        epochs = self.config["epoch"]        
        
        no_update = 0
        
        for epoch in range(1,epochs+1):
            self.train_loop(epoch)
            
            val_loss = self.valid_loop(epoch)

            if val_loss < min_loss:
                no_update = 0
                print("Save model at epoch {}".format(epoch))
                self.save_model()
            else:
                no_update += 1
                
            if no_update > 30:
                print("Early Stop...")
                break
    
            self.scheduler.step(epoch)
        
            
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
            
            total_output = np.append(total_output, model_output.cpu().detach().numpy(), axis = 0)
            total_label = np.append(total_label, label.cpu().detach().numpy(), axis = 0)
            
            if step % 100 == 0:
                print("[{}] STEP : ({}/{}) | TRAIN LOSS : {} | Time : {}]".format(
                epoch, step, steps, np.mean(tmp_train_loss), time.time()-prev_time))
                sys.stdout.flush()
                prev_time = time.time()
            step += 1
        
        f1_score = self.calculation(total_output, total_label)
        total_train_loss = np.array(total_train_loss)
        loss = np.mean(total_train_loss)

        self.logs["train_f1_score"] = f1_score
        self.logs["train_loss"] = loss
        print("F1_Score : {} | Loss : {} ".format(round(f1_score, 5), round(loss, 5)))

        if self.enable_log:
            lr = self.scheduler.get_lr()
            self.neptune["learning_rate"].log(lr)
            self.neptune["train/f1_score"].log(f1_score)
            self.neptune["train/loss"].log(loss)
        

    def valid_loop(self, epoch):
        self.model.eval()
        
        total_valid_loss = []
        tmp_valid_loss = []
        
        step = 1
        steps = len(self.valid_loader)
        prev_time = time.time()
        
        total_output = np.empty((0,10))
        total_label = np.empty((0,10))
        
        with torch.no_grad():
            for img, label in iter(self.valid_loader):
                img, label = img.float().to(self.device), label.float().to(self.device)
                self.optimizer.zero_grad()
                
                model_output = self.model(img)
                
                loss = self.criterion(model_output.squeeze(1), label)
                # loss.backward()
                # self.optimizer.step()
                    
                tmp_valid_loss.append(loss.item())
                total_valid_loss.append(loss.item())
                
                total_output = np.append(total_output, model_output.cpu().detach().numpy(), axis = 0)
                total_label = np.append(total_label, label.cpu().detach().numpy(), axis = 0)
                
                if step % 100 == 0:
                    print("[{}] STEP : ({}/{}) | VALID LOSS : {} | Time : {}]".format(
                    epoch, step, steps, np.mean(tmp_valid_loss), time.time()-prev_time))
                    sys.stdout.flush()
                    prev_time = time.time()
                step += 1
        
        f1_score = self.calculation(total_output, total_label)
        total_valid_loss = np.array(total_valid_loss)
        loss = np.mean(total_valid_loss)
        
        self.logs["valid_f1_score"] = f1_score
        self.logs["valid_loss"] = loss
        print("F1_Score : {} | Loss : {} ".format(round(f1_score, 5), round(loss, 5)))
        
        if self.enable_log:
            self.neptune["valid/f1_score"].log(f1_score)
            self.neptune["valid/loss"].log(loss)
        
        return loss
    
    def test(self):
        self.model.eval()
        
        total_test_loss = []
        tmp_test_loss = []
        
        step = 1
        steps = len(self.test_loader)
        prev_time = time.time()
        
        total_output = np.empty((0,10))
        total_label = np.empty((0,10))
        
        with torch.no_grad():
            for img, label in iter(self.test_loader):
                img, label = img.float().to(self.device), label.float().to(self.device)
                self.optimizer.zero_grad()
                
                model_output = self.model(img)
                
                loss = self.criterion(model_output.squeeze(1), label)
                    
                tmp_test_loss.append(loss.item())
                total_test_loss.append(loss.item())
                
                total_output = np.append(total_output, model_output.cpu().detach().numpy(), axis = 0)
                total_label = np.append(total_label, label.cpu().detach().numpy(), axis = 0)
                
                if step % 100 == 0:
                    print("STEP : ({}/{}) | TEST LOSS : {} | Time : {}]".format(
                    step, steps, np.mean(tmp_test_loss), time.time()-prev_time))
                    sys.stdout.flush()
                    prev_time = time.time()
                step += 1
        
        print("* TEST RESULT *")
        f1_score = self.calculation(total_output, total_label)
        total_valid_loss = np.array(total_valid_loss)
        loss = np.mean(total_valid_loss)
        print("F1_Score : {} | Loss : {} ".format(round(f1_score, 5), round(loss, 5)))
        
    def calculation(self, output_list, label_list):        
        output_list = np.argmax(output_list, axis = 1)
        label_list = np.argmax(label_list, axis = 1)
        
        class_list = []
        for class_name in CLASS_NUMBER:
            class_list.append(class_name)
        
        # acc = accuracy_score(label_list, output_list, average = None)
        prec = precision_score(label_list, output_list, average = None)
        rec = recall_score(label_list, output_list, average = None)
        
        f1_total = f1_score(label_list, output_list, average = 'macro')
        f1_s = f1_score(label_list, output_list, average=None)
        
        # print("=== Accuracy List ===")
        # for i, class_ in enumerate(class_list):
        #     print("{} | {}".format(class_, round(acc[i], 5)))
            
        print("=== Precision | Recall | F1 Score ===")
        for i, class_ in enumerate(class_list):
            print("{} |{}|{}|{}|".format(class_, round(prec[i], 5), round(rec[i], 5), round(f1_s[i], 5)))  
    
        print("Total F1 : {}".format(f1_total))
        
        # metric = metrics.classification_report(label_list, output_list, digits=10, target_names=class_list)
        # print(metric)
        
        return f1_total
    
    def save_model(self):
        save_path = os.path.abspath(self.config["save_path"])
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
            
        model_name = self.config["model_name"]
        model_path = os.path.join(save_path, model_name + ".pt")
        
        if self.multi_gpu:
            torch.save(self.model.module, model_path)
        else:
            torch.save(self.model, model_path)