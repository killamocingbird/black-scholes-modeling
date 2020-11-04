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

from model import BS_PDE, L2, Burgers_PDE
import modules as m
import util

import warnings
warnings.filterwarnings("error")

# Hyperparameters for run
header = 'BS_1_fixed_'    

device = 'cuda' if torch.cuda.is_available() else 'cpu'
debug = False
seed = 1                               # Seeds for stochastic reproducability
batch_size = 500                       # Batch size for SGD
epochs = 500                           # Number of epochs to train network for
lr = 1e-1                               # Initial learning rate
#checkpoint = header + 'checkpoint.pth'
checkpoint = None
verbose = 10                           # Number of epochs to print, -1 for none

# Set seed
util.set_seed(seed, device)

# Load in data
dataset = scipy.io.loadmat('burgers_shock.mat')
t = dataset['t'].flatten()[:,None]
x = dataset['x'].flatten()[:,None]
Exact = np.real(dataset['usol']).T

X, T = np.meshgrid(x,t)
xdata = torch.as_tensor(np.hstack((X.flatten()[:,None], T.flatten()[:,None]))).to(device).float()
ydata = torch.as_tensor(Exact.flatten()[:,None]).to(device).float()

# 90 / 10 split on training and validation
xtrain = xdata[:9 * len(xdata) // 10]
ytrain = ydata[:9 * len(xdata) // 10]

xval = xdata[9 * len(xdata) // 10:]
yval = ydata[9 * len(xdata) // 10:]

# Declare model and optimizer
model = Burgers_PDE(device)
model = model.to(device)
optimizer = optim.LBFGS(model.parameters(), lr=lr, history_size=20, line_search_fn='strong_wolfe')

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
        h, V = model.pde_structure(xtrain)
        loss = L2(h, torch.zeros(h.shape).to(device)) + \
               L2(V.squeeze(), ytrain.squeeze())
        optimizer.zero_grad()
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    running_loss = closure()

    model.eval()
    h, V = model.pde_structure(xval)
    running_val_loss = L2(h, torch.zeros(h.shape).to(device)) + \
                       L2(V.squeeze(), yval.squeeze())

    model.update_loss(running_loss, val_loss=running_val_loss)
    if verbose != -1 and (epoch+1)%verbose == 0:
        print("[%d] train: %.8f | val: %.8f | l1: %.5f | l2: %.5f | elapsed: %.2f (s)" % (epoch+1, running_loss, running_val_loss, model.lambda_1.item(), torch.exp(model.lambda_2).item(), (time.time()-start_time)*verbose))
        time.sleep(0.5)
    
    if running_val_loss < min_loss:
        min_loss = running_val_loss
        model.save(header=header, optimizer=optimizer)
        
    #sigmas.append(model.sigma.detach().cpu().item())

#plt.plot([i for i in range(1, len(sigmas) + 1)], sigmas)
#plt.xlabel('Epoch')
#plt.ylabel('Implied Volatility')

