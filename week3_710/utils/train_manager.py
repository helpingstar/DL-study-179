import torch
import torch.nn as nn
import numpy as np
import os
from sklearn import metrics

class TrainManager():
    def __init__(self, model, device, dataloader, config, neptune_instance) -> None:
        self.model = model
        self.device = device
        self.config = config 
        self.dataloader = dataloader
        self.optimizer = torch.optim.Adam(model.parameters(), lr = config["learning_rate"])
        
        self.neptune = neptune_instance
    
    
    def train(self):
        self.model.to(self.device)
        
        epochs = self.config["epochs"]
        min_loss = 9999
        no_update = 0
        
        for i in range(1, epochs+1):
            train_loss = self.train_loop(i)
    
            valid_loss = self.valid_loop(i)
            
            if min_loss > valid_loss:
                no_update = 0 
                print("Save Best Model")
                self.save_model()
            else:
                no_update += 1
            
            if no_update > 20:
                print("Early Stop at epoch ", str(i))
                break
            

    def train_loop(self, epoch):
        self.model.train()
        
        ## Train
        print("Train Epoch : {}".format(epoch))
        step = 1
        total_step = len(self.dataloader["train"])
        
        losses = []
        
        for data in iter(self.dataloader["train"]):
            self.optimizer.zero_grad()

            user = data["user"].to(self.device)
            movie = data["movie"].to(self.device)
            rating = data["rating"].to(self.device).view(-1, 1)
            output = self.model(user, movie)
            
            loss = nn.MSELoss()(output.to(torch.float32), rating.to(torch.float32))
            loss.backward()
            self.optimizer.step()

            temp_loss = loss.cpu().detach().numpy()
            losses.append(temp_loss)
            
            if step % 100 == 0:
                rms = self.monitor(output, rating)
                print("[{}|{}] Loss : {} | RMS : {}".format(step, total_step, temp_loss, rms))
            step += 1

        mean_loss = np.mean(np.array(losses))
        print("Train Loss : {}".format(mean_loss))
        self.neptune["train/loss"].log(mean_loss)
        
        return mean_loss
    
    def valid_loop(self, epoch):
        self.model.eval()
        
        ## Valid
        with torch.no_grad():
            print("Valid Epoch : {}".format(epoch))
            step = 1
            total_step = len(self.dataloader["valid"])
            
            losses = []
            
            for data in iter(self.dataloader["valid"]):
                user = data["user"].to(self.device)
                movie = data["movie"].to(self.device)
                rating = data["rating"].to(self.device).view(-1, 1)
                
                output = self.model(user, movie)
                
                loss = nn.MSELoss()(output, rating)
                temp_loss = loss.cpu().detach().numpy()
                losses.append(temp_loss)
                
                if step % 100 == 0:
                    rms = self.monitor(output, rating)
                    print("[{}|{}] Loss : {} | RMS : {}".format(step, total_step, temp_loss, rms))
                    print("Predict Output")
                    print(output.view(-1).cpu().detach().numpy()[:5])
                    print("Rating")                    
                    print(rating.view(-1).cpu().detach().numpy()[:5])
                    print("_________________")
                
                step += 1
        
        mean_loss = np.mean(np.array(losses))
        print("Valid Loss : {}".format(mean_loss))
        self.neptune["valid/loss"].log(mean_loss)

        return mean_loss
 
 
    def monitor(self, output, rating):
        output = output.detach().cpu().numpy()
        rating = rating.detach().cpu().numpy()
        
        return np.sqrt(metrics.mean_squared_error(rating, output))   


    def save_model(self):
        save_path = os.path.abspath(self.config["save_path"])
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
            
        model_name = self.config["model_name"]
        model_path = os.path.join(save_path, model_name + ".pt")
        
        torch.save(self.model, model_path)