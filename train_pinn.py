import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from model import BS_PDE, BS_SDE, L2
import option_pricing_BS as bs
import modules as m
import util

# Hyperparameters for run
header = 'BS_1_fixed_sigma_'    

device = 'cuda' if torch.cuda.is_available() else 'cpu'
debug = False
seed = 100                               # Seeds for stochastic reproducability
batch_size = 1024                       # Batch size for SGD
epochs = 2000                           # Number of epochs to train network for
lr = 5e-4                               # Initial learning rate
#checkpoint = header + 'checkpoint.pth'
checkpoint = None
verbose = 1                           # Number of epochs to print, -1 for none

### Strike price and interest rate respectively ###
k = 1
r = 0.0

# Set seed
util.set_seed(seed, device)

# Load in data
dataset = util.import_dataset(data_path='option_call_dataset_simplified.txt')
# dataset = util.import_dataset(data_path='option_call_dataset_brownian.txt')
# Shuffle
# dataset = torch.as_tensor(dataset[np.random.permutation(len(dataset))], device=device).float()
dataset = torch.as_tensor(dataset, device=device).float()

# 90 / 10 split on training and validation
xtrain = dataset[:, :-1]
ytrain = dataset[:, -1]
theta = bs.theta(xtrain[:,0].cpu().numpy(), k, xtrain[:,2].cpu().numpy(), r, 0.65)
theta = torch.as_tensor(theta, device=device).float()
gamma = bs.gamma(xtrain[:,0].cpu().numpy(), k, xtrain[:,2].cpu().numpy(), r, 0.65)
gamma = torch.as_tensor(gamma, device=device).float()

xval = dataset[95 * len(dataset) // 100:, :-1]
yval = dataset[95 * len(dataset) // 100:, -1]

# Drop everything except for S and T
xtrain = torch.cat((xtrain[:,0:1], xtrain[:,2:3]), 1)
xval = torch.cat((xval[:,0:1], xval[:,2:3]), 1)

##### Sort based on time to get SDE data #####
def sde_setup(x, y):
    # Add y as last column for full dataset
    data = torch.cat((x, y.unsqueeze(1)), 1)
    
    # Extract times
    times = data[:,1].unique()
    times.sort()
    
    # Put all datapoints from the same time into groups
    pools = [data[data[:,1]==t] for t in times]
    # For each entree in each pool, generate an index to pair from the previous pool
    selects = [torch.as_tensor([random.randint(0, len(pools[i])-1) 
                                for j in range(len(pools[i+1]))]).to(x.device)
               for i in range(len(pools)-1)]
    # Create newly paired data
    ret = torch.ones(0, 6).to(x.device)
    
    for i in range(1, len(pools)):
        # Pair pool with previous data
        to_append = torch.cat((pools[i], pools[i-1][selects[i-1]]), 1)
        ret = torch.cat((ret, to_append), 0)
    
    return ret

# train_data = sde_setup(xtrain, ytrain)
# val_data = sde_setup(xval, yval)
train_data = torch.cat((xtrain[1:], ytrain.unsqueeze(1)[1:], xtrain[:-1], ytrain.unsqueeze(1)[:-1]), 1)
val_data = torch.cat((xval[1:], yval.unsqueeze(1)[1:], xval[:-1], yval.unsqueeze(1)[:-1]), 1)

### Temp loading
#dataset = torch.load('brownian_dataset_500_500.pth').to(device).float()
#train_data = dataset[:9 * len(dataset) // 10]
#val_data = dataset[9 * len(dataset) // 10:]

###

# Declare model and optimizer
#model = BS_PDE(device)
model = BS_SDE(device)
model = model.to(device)
#optimizer = optim.Adam([
#        {'params': model.layers.parameters(), 'weight_decay':0},
#        {'params': model.sigma, 'lr': 1e-3}
#    ], lr=lr)
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
    running_loss = 0
    # Shuffle data
    perm = np.random.permutation(len(train_data))
    train_data = train_data[perm]
    theta = theta[perm]
    gamma = gamma[perm]
    
    xtrain = train_data[:,:2]
    ytrain = train_data[:,2]
    xval = val_data[:,:2]
    yval = val_data[:,2]
    
    model.train()
    train_p_bar = tqdm(range(len(xtrain) // batch_size))
    for batch_idx in train_p_bar:
        idx = slice(batch_idx*batch_size,(batch_idx+1)*batch_size)
        xbatch = xtrain[idx]
        ybatch = ytrain[idx]
        tbatch = theta[idx]
        gbatch = gamma[idx]
        prev_batch = train_data[idx][:,3:-1]
        
        h, V = model.pde_structure(xbatch[:,:3], r)
        loss = L2(h, torch.zeros(h.shape).to(device)) + \
               model.boundary_conditions(xbatch[:,:3], k, r, epoch=epoch) + \
               1e-1 * L2(tbatch, model.get_theta(xbatch[:,:3])) + \
               5e-2 * L2(gbatch, model.get_gamma(xbatch[:,:3]))
               #L2(V.squeeze(), ybatch) + \
               #model.sde_conditions(xbatch, prev_batch, r)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_p_bar.set_description("Loss %.6f" % loss.item())
    running_loss /= (batch_idx + 1)

    running_val_loss = 0  
    model.eval()
    for batch_idx in range(len(xval) // batch_size):
        idx = slice(batch_idx*batch_size,(batch_idx+1)*batch_size)
        xbatch = xval[idx]
        ybatch = yval[idx]
        prev_batch = val_data[idx][:,3:-1]
    
        h, V = model.pde_structure(xbatch[:,:3], r)
        loss = L2(h, torch.zeros(h.shape).to(device)) + \
               L2(V.squeeze(), ybatch) + \
               model.boundary_conditions(xbatch[:,:3], k, r, epoch=epoch)
               #model.sde_conditions(xbatch, prev_batch, r)
               
        running_val_loss += loss.item()
    running_val_loss /= (batch_idx + 1)
    model.update_loss(running_loss, val_loss=running_val_loss)
    if verbose != -1 and epoch%verbose == 0:
        print("[%d] train: %.8f | val: %.8f | sigma: %.3f |  elapsed: %.2f (s)" % (epoch+1, running_loss, running_val_loss, model.sigma, (time.time()-start_time)*verbose))
        time.sleep(0.5)
    
    if running_val_loss < min_loss:
        min_loss = running_val_loss
        model.save(header=header, optimizer=optimizer)
        
    #sigmas.append(model.sigma.detach().cpu().item())
    sigmas.append(model.sigma)
    
bs.evaluate(model, dataset)
    