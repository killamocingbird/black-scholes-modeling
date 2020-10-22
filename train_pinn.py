"""
Main training script
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from model import BS_PDE, L2
import modules as m
import util

# Hyperparameters for run
header = 'BS_1_fixed_'    

device = 'cuda' if torch.cuda.is_available() else 'cpu'
debug = False
seed = 1                               # Seeds for stochastic reproducability
batch_size = 500                       # Batch size for SGD
epochs = 500                           # Number of epochs to train network for
lr = 5e-4                               # Initial learning rate
#checkpoint = header + 'checkpoint.pth'
checkpoint = None
verbose = 1                           # Number of epochs to print, -1 for none

# Set seed
util.set_seed(seed, device)

# Load in data
dataset = util.import_dataset(data_path='option_call_dataset_frozen.txt')
# Shuffle
dataset = torch.as_tensor(dataset[np.random.permutation(len(dataset))], device=device).float()

# 90 / 10 split on training and validation
xtrain = dataset[:9 * len(dataset) // 10, :-1]
ytrain = dataset[:9 * len(dataset) // 10, -1]

xval = dataset[9 * len(dataset) // 10:, :-1]
yval = dataset[9 * len(dataset) // 10:, -1]

# Declare model and optimizer
model = BS_PDE(device)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Load in checkpoint
if checkpoint is not None:
    print("Loading checkpoint")
    checkpoint = torch.load(checkpoint)
    model.load(checkpoint)
    model = model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

min_loss = 1e10
sigmas = []
print("Optimizing %d parameters on %s" % (util.count_parameters(model), device))
time.sleep(0.5)
for epoch in range(epochs):
    # Begin timer
    start_time = time.time()
    
    # Shuffle training data
    perm = np.random.permutation(len(xtrain))
    xtrain = xtrain[perm]
    ytrain = ytrain[perm]
    running_loss = 0
    model.train()
    train_p_bar = tqdm(range(len(xtrain) // batch_size))
    for batch_idx in train_p_bar:
        xbatch = xtrain[batch_idx*batch_size:(batch_idx+1)*batch_size]
        ybatch = ytrain[batch_idx*batch_size:(batch_idx+1)*batch_size]
        
        h, V = model.pde_structure(xbatch[:,:3], xbatch[:,3])
        
        loss = L2(h, torch.zeros(h.shape).to(device)) + \
               model.boundary_conditions(xbatch[:,:3], xbatch[:,3], epoch=epoch) + \
               L2(V.squeeze(), ybatch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_p_bar.set_description("Loss %.6f" % loss.item())
    running_loss /= (batch_idx + 1)

    running_val_loss = 0  
    model.eval()
    for batch_idx in range(len(xval) // batch_size):
        xbatch = xval[batch_idx*batch_size:(batch_idx+1)*batch_size]
        ybatch = yval[batch_idx*batch_size:(batch_idx+1)*batch_size]
    
        h, V = model.pde_structure(xbatch[:,:3], xbatch[:,3])
    
        loss = L2(h, torch.zeros(h.shape).to(device)) + \
               model.boundary_conditions(xbatch[:,:3], xbatch[:,3]) + \
               L2(V.squeeze(), ybatch)
               
        running_val_loss += loss.item()
    running_val_loss /= (batch_idx + 1)
    model.update_loss(running_loss, val_loss=running_val_loss)
    if verbose != -1 and epoch%verbose == 0:
        print("[%d] train: %.8f | val: %.8f | sigma: %.3f |  elapsed: %.2f (s)" % (epoch+1, running_loss, running_val_loss, model.sigma.item(), (time.time()-start_time)*verbose))
        time.sleep(0.5)
    
    if running_val_loss < min_loss:
        min_loss = running_val_loss
        model.save(header=header, optimizer=optimizer)
        
    sigmas.append(model.sigma.detach().cpu().item())

plt.plot([i for i in range(1, len(sigmas) + 1)], sigmas)
plt.xlabel('Epoch')
plt.ylabel('Implied Volatility')

