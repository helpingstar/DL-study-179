import torch
import torch.nn as nn


class RecModel(nn.Module):
    def __init__(self, num_user, num_movie):
        super().__init__()
        self.user_embed = nn.Embedding(num_user, 64)
        self.movie_embed = nn.Embedding(num_movie, 64)
    
        self.attention = nn.MultiheadAttention(embed_dim = 128, num_heads = 4, kdim = 128, vdim = 128, batch_first=True)
        self.out = nn.Linear(128, 1)
    
    
    def forward(self, users, movies):
        user_embeds = self.user_embed(users)
        movie_embeds = self.movie_embed(movies)
        
        concat_embeds = torch.cat([user_embeds, movie_embeds], dim = 1)
        output, _ = self.attention(concat_embeds, concat_embeds, concat_embeds)
        
        output = self.out(output)
        
        return output
    