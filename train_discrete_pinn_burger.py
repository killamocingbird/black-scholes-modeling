import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import time
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from model import L2, Discrete_Burgers_PDE
import modules as m
import util

import warnings
warnings.filterwarnings("error")

# Hyperparameters for run
header = 'Burger_1_fixed_discrete_'    

device = 'cuda' if torch.cuda.is_available() else 'cpu'
debug = False
seed = 1                               # Seeds for stochastic reproducability
batch_size = 500                       # Batch size for SGD
epochs = 500                           # Number of epochs to train network for
lr = 1e-2                               # Initial learning rate
#checkpoint = header + 'checkpoint.pth'
checkpoint = None
verbose = 5                           # Number of epochs to print, -1 for none

# Set seed
util.set_seed(seed, device)

# Load in data
data = scipy.io.loadmat('burgers_shock.mat')
t_star = data['t'].flatten()[:,None]
x_star = data['x'].flatten()[:,None]
Exact = np.real(data['usol'])
    
idx_t = 10

noise = 0.0    
N0 = 199
N1 = 201
skip = 80

idx_x = np.random.choice(Exact.shape[0], N0, replace=False)
x0 = x_star[idx_x,:]
u0 = Exact[idx_x,idx_t][:,None]
    
idx_x = np.random.choice(Exact.shape[0], N1, replace=False)
x1 = x_star[idx_x,:]
u1 = Exact[idx_x,idx_t + skip][:,None]

dt = (t_star[idx_t+skip] - t_star[idx_t]).item()
RK_order = int(np.ceil(0.5*np.log(np.finfo(float).eps)/np.log(dt)))

x0 = torch.as_tensor(x0, device=device).float()
x1 = torch.as_tensor(x0, device=device).float()
u0 = torch.as_tensor(x0, device=device).float().squeeze()
u1 = torch.as_tensor(x0, device=device).float().squeeze()

# 90 / 10 split on training and validation
xtrain0 = x0[:9 * len(x0) // 10]
ytrain0 = u0[:9 * len(u0) // 10]
xval0 = x0[9 * len(x0) // 10:]
yval0 = u0[9 * len(u0) // 10:]

xtrain1 = x1[:9 * len(x0) // 10]
ytrain1 = u1[:9 * len(u0) // 10]
xval1 = x1[9 * len(x0) // 10:]
yval1 = u1[9 * len(u0) // 10:]

# Declare model and optimizer
model = Discrete_Burgers_PDE(dt, RK_order, device)
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
    #perm = np.random.permutation(len(xtrain))
    #xtrain = xtrain[perm]
    #ytrain = ytrain[perm]
    
    model.train()
    def closure():
        loss = (L2(model.get_u_0(xtrain0), ytrain0.expand(RK_order, len(ytrain0)).t()) + 
                L2(model.get_u_1(xtrain1), ytrain1.expand(RK_order, len(ytrain1)).t()))
        optimizer.zero_grad()
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    running_loss = closure()

    model.eval()
    running_val_loss = (L2(model.get_u_0(xval0), yval0.expand(RK_order, len(yval0)).t()) + 
                        L2(model.get_u_1(xval1), yval1.expand(RK_order, len(yval1)).t()))

    model.update_loss(running_loss, val_loss=running_val_loss)
    if verbose != -1 and (epoch+1)%verbose == 0:
        print("[%d] train: %.8f | val: %.8f | l1: %.5f | l2: %.5f | elapsed: %.2f (s)" % (epoch+1, running_loss, running_val_loss, model.lambda_1.item(), torch.exp(model.lambda_2).item(), (time.time()-start_time)*verbose))
        time.sleep(0.5)
    
    if running_val_loss < min_loss:
        min_loss = running_val_loss
        model.save(header=header, optimizer=optimizer)
        


































