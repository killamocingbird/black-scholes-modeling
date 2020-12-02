import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from model import Discrete_BS_PDE, L2
import modules as m
import util

# Hyperparameters for run
header = 'BS_1_fixed_discrete_'    

device = 'cuda' if torch.cuda.is_available() else 'cpu'
debug = False
seed = 10000                               # Seeds for stochastic reproducability
batch_size = 16                       # Batch size for SGD
epochs = 500                           # Number of epochs to train network for
lr = 1e-3                               # Initial learning rate
#checkpoint = header + 'checkpoint.pth'
checkpoint = None
verbose = 10                           # Number of epochs to print, -1 for none
RK_order = 7

### Strike price and interest rate respectively ###
k = 1
r = 0.03

# Set seed
util.set_seed(seed, device)

# Load in data
dataset = util.import_dataset(data_path='option_call_dataset_frozen.txt')
# Shuffle
dataset = torch.as_tensor(dataset[np.random.permutation(len(dataset))], device=device).float()

# Pick two distinct time points
d0 = dataset[dataset[:,2] == dataset[:,2].unique()[50]]
d1 = dataset[dataset[:,2] == dataset[:,2].unique()[-50]]

# 90 / 10 split on training and validation
xtrain0 = d0[:9 * len(d0) // 10, :-1]
ytrain0 = d0[:9 * len(d0) // 10, -1]
xtrain1 = d1[:9 * len(d1) // 10, :-1]
ytrain1 = d1[:9 * len(d1) // 10, -1]

xval0 = d0[9 * len(d0) // 10:, :-1]
yval0 = d0[9 * len(d0) // 10:, -1]
xval1 = d1[9 * len(d1) // 10:, :-1]
yval1 = d1[9 * len(d1) // 10:, -1]

# Drop everything except for S and T
xtrain0 = torch.cat((xtrain0[:,0:1], xtrain0[:,2:3]), 1)
xval0 = torch.cat((xval0[:,0:1], xval0[:,2:3]), 1)

xtrain1 = torch.cat((xtrain1[:,0:1], xtrain1[:,2:3]), 1)
xval1 = torch.cat((xval1[:,0:1], xval1[:,2:3]), 1)

assert len(xtrain0) == len(xtrain1)
assert len(xval0) == len(xval1)

# Declare model and optimizer
model = Discrete_BS_PDE(dataset[:,2].unique()[-50]-dataset[:,2].unique()[50], RK_order, device)
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
    perm = np.random.permutation(len(xtrain0))
    xtrain0 = xtrain0[perm]
    ytrain0 = ytrain0[perm]
    xtrain1 = xtrain1[perm]
    ytrain1 = ytrain1[perm]
    running_loss = 0
    model.train()
    for batch_idx in range(len(xtrain0) // batch_size):
        xbatch0 = xtrain0[batch_idx*batch_size:(batch_idx+1)*batch_size]
        ybatch0 = ytrain0[batch_idx*batch_size:(batch_idx+1)*batch_size]
        
        xbatch1 = xtrain1[batch_idx*batch_size:(batch_idx+1)*batch_size]
        ybatch1 = ytrain1[batch_idx*batch_size:(batch_idx+1)*batch_size]
        
        loss = (L2(model.get_u_0(xbatch0, r), ybatch0.expand(RK_order, len(ybatch0)).t()) + 
                L2(model.get_u_1(xbatch1, r), ybatch1.expand(RK_order, len(ybatch1)).t()))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    running_loss /= (batch_idx + 1)

    running_val_loss = 0  
    model.eval()
    for batch_idx in range(len(xval0) // batch_size):
        xbatch0 = xval0[batch_idx*batch_size:(batch_idx+1)*batch_size]
        ybatch0 = yval0[batch_idx*batch_size:(batch_idx+1)*batch_size]
        
        xbatch1 = xval1[batch_idx*batch_size:(batch_idx+1)*batch_size]
        ybatch1 = yval1[batch_idx*batch_size:(batch_idx+1)*batch_size]
    
        loss = (L2(model.get_u_0(xbatch0, r), ybatch0.expand(RK_order, len(ybatch0)).t()) + 
                L2(model.get_u_1(xbatch1, r), ybatch1.expand(RK_order, len(ybatch1)).t()))
               
        running_val_loss += loss.item()
    running_val_loss /= (batch_idx + 1)
    model.update_loss(running_loss, val_loss=running_val_loss)
    if verbose != -1 and (epoch+1)%verbose == 0:
        print("[%d] train: %.8f | val: %.8f | sigma: %.3f |  elapsed: %.2f (s)" % (epoch+1, running_loss, running_val_loss, model.sigma.item(), (time.time()-start_time)*verbose))
        time.sleep(0.5)
    
    if running_val_loss < min_loss:
        min_loss = running_val_loss
        model.save(header=header, optimizer=optimizer)
        
    sigmas.append(model.sigma.detach().cpu().item())

plt.plot([i for i in range(1, len(sigmas) + 1)], sigmas)
plt.plot([i for i in range(1, len(sigmas) + 1)], [0.65 for i in range(1, len(sigmas) + 1)], '--')
plt.legend(['Learned', 'Actual'])
plt.xlabel('Epoch')
plt.ylabel('Implied Volatility')

