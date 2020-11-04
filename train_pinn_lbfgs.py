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
lr = 1e-1                               # Initial learning rate
#checkpoint = header + 'checkpoint.pth'
checkpoint = None
verbose = 5                           # Number of epochs to print, -1 for none

# Set seed
util.set_seed(seed, device)

# Load in data
dataset = util.import_dataset(data_path='option_call_dataset_frozen.txt')
# Shuffle
dataset = torch.as_tensor(dataset[np.random.permutation(len(dataset))], device=device).float()
dataset = dataset[:10000]

# 90 / 10 split on training and validation
xtrain = dataset[:9 * len(dataset) // 10, :-1]
ytrain = dataset[:9 * len(dataset) // 10, -1]

xval = dataset[9 * len(dataset) // 10:, :-1]
yval = dataset[9 * len(dataset) // 10:, -1]

# Declare model and optimizer
model = BS_PDE(device)
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
    perm = np.random.permutation(len(xtrain))
    xtrain = xtrain[perm]
    ytrain = ytrain[perm]
    
    model.train()
    def closure():
        h, V = model.pde_structure(xtrain[:,:3], xtrain[:,3])
        loss = L2(h, torch.zeros(h.shape).to(device)) + \
               model.boundary_conditions(xtrain[:,:3], xtrain[:,3], epoch=epoch) + \
               L2(V.squeeze(), ytrain)
        optimizer.zero_grad()
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    running_loss = closure()

    model.eval()
    h, V = model.pde_structure(xval[:,:3], xval[:,3])
    running_val_loss = L2(h, torch.zeros(h.shape).to(device)) + \
                       model.boundary_conditions(xval[:,:3], xval[:,3]) + \
                       L2(V.squeeze(), yval)

    model.update_loss(running_loss, val_loss=running_val_loss)
    if verbose != -1 and (epoch+1)%verbose == 0:
        print("[%d] train: %.8f | val: %.8f | sigma: %.3f |  elapsed: %.2f (s)" % (epoch+1, running_loss, running_val_loss, model.sigma.item(), (time.time()-start_time)*verbose))
        time.sleep(0.5)
    
    if running_val_loss < min_loss:
        min_loss = running_val_loss
        model.save(header=header, optimizer=optimizer)
        
    sigmas.append(model.sigma.detach().cpu().item())

plt.plot([i for i in range(1, len(sigmas) + 1)], sigmas)
plt.xlabel('Epoch')
plt.ylabel('Implied Volatility')

