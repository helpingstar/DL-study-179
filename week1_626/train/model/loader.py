import os
import torch

def load_model(model, model_path, model_name, device):
    model_path = os.path.join(model_path,model_name + ".pt")
    model_path = os.path.abspath(model_path)
    print(model_path)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)

    return model