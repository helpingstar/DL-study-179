import torch
import neptune.new as neptune

from utils.dataset import MovieDataset
from utils.reader import get_data, get_train_valid
from torch.utils.data import DataLoader
from model.multiattention import RecModel
from utils.train_manager import TrainManager

     
CFG = {
    "save_path" : "./saved",
    "epochs" : 100,
    "learning_rate" : 1e-3,
    "model_name" : "MHA"
}





if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available else "cpu"
    
    neptune_instance = neptune.init(
        project="hsh-dev/dl-study",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZDNlMTkxZS0wOTE4LTRhYzUtODUzNS1hNGUyOTkzMTU0MjgifQ==",
    )
    neptune_instance["parameters"] = CFG
    
    
    
    df_train, df_valid, max_user, max_movie = get_train_valid()

    
    train_dataset = MovieDataset(df_train.userId.values, 
                                 df_train.movieId.values, 
                                 df_train.rating.values)
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle=True)
    
    valid_dataset = MovieDataset(df_valid.userId.values,
                                 df_valid.movieId.values,
                                 df_valid.rating.values)
    valid_loader = DataLoader(valid_dataset, batch_size = 64, shuffle=False)


    model = RecModel(num_user = max_user+1, num_movie = max_movie+1)
    
    dataloader = {}
    dataloader["train"] = train_loader
    dataloader["valid"] = valid_loader
    
    train_manager = TrainManager(model, device, dataloader, CFG, neptune_instance)
    train_manager.train()
    