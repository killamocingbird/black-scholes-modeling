import csv
import numpy as np
import torch
import os


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def import_dataset(data_path='option_call_dataset.txt'):
    assert os.path.isfile(data_path), "Data file does not exist"
    return np.loadtxt(data_path, delimiter=',')
    

def set_seed(seed, device=None):
    import numpy as np
    import torch
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
