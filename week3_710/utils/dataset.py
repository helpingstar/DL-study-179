import torch
import torch.nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MovieDataset(Dataset):
    def __init__(self, users, movies, ratings) -> None:
        self.users = users
        self.movies = movies
        self.ratings = ratings
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, index):
        user = self.users[index]
        movie = self.movies[index]
        rating = self.ratings[index]
        
        return {"user" : torch.tensor(user, dtype = torch.long),
                "movie" : torch.tensor(movie, dtype = torch.long),
                "rating" : torch.tensor(rating, dtype = torch.long)}
