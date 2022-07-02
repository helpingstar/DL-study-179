import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_attention(query, key, value):
    # query, key --> B x N x K
    # value --> B x N x V
    
    d_k = key.size(-1)
    
    A = torch.einsum('bch,bth->bct',[query, key])   # B x N x N
    A = F.softmax(A * d_k ** (-0.5), 2)    
    R = torch.einsum('bhw,bwl->bhl', [A, value])    # B x N x V
    
    return R


class SelfAttention(nn.Module):
    def __init__(self, d_embedding, d_model, d_key, d_value):
        super().__init__()
        
        self.N = d_embedding
        self.M = d_model
        self.K = d_key
        self.V = d_value
        
        self.W_query = nn.Linear(d_model, d_key)
        self.W_key = nn.Linear(d_model, d_key)
        self.W_value = nn.Linear(d_model, d_value)
        

    def forward(self, x):
        # x shape : B x N(d_embedding) x M(d_model)
        query = self.W_query(x) # B x N x K
        key = self.W_key(x)     # B x N x K
        value = self.W_value(x) # B x N x V
        
        A = calculate_attention(query, key, value)
        return A
    

if __name__ == "__main__":
    # input size : B x N x M
    input = torch.randn(5, 3, 4)
    print("Input : {}".format(input.size()))
    model = SelfAttention(d_embedding = 3, d_model = 4, d_key = 5, d_value = 6)
    
    # output size : B x N x V
    output = model(input)    
    print("Output : {}".format(output.size()))
