import matplotlib.pyplot as plt
import numpy as np
import torch

from model import BS_PDE, L2
import option_pricing_BS as bs
import util


checkpoint = 'BS_1_fixed_sigma_checkpoint.pth'    

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load in data
dataset = torch.as_tensor(util.import_dataset(data_path='option_call_dataset_simplified.txt')).to(device).float()

# Declare model and optimizer
model = BS_PDE(device)
model = model.to(device)

# Load in checkpoint
if checkpoint is not None:
    print("Loading checkpoint")
    checkpoint = torch.load(checkpoint)
    model.load(checkpoint)
    model = model.to(device)

k = dataset[:,1]
dataset = torch.cat((dataset[:,0:1], dataset[:,2:]), 1)
pred_value = model(dataset[:,:2])

# Colored scattered plot in 3D
def scatter_col(x, y, z, n, ax):
    partitions = np.linspace(z.min(), z.max(), n+1)
    
    temp = np.array([0.5 for i in range(len(partitions))])
    colors = np.stack([(partitions - z.min().numpy())/(z.max().numpy() - z.min().numpy()), temp, temp]).transpose()
    
    for i in range(n):
        filt = (partitions[i] <= z) * (z < partitions[i + 1])
        ax.scatter(x[filt], y[filt], z[filt], c=colors[i])
exit()
# Generate and save colored scatter plot of model and real data respectivelys
def scatter_both(model_pred, gt, z_label, save_path):
    fig = plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(1, 2, 1, projection='3d')
    scatter_col(model_pred[0], model_pred[1], model_pred[2], 10, ax1)
    ax1.set_title("Model output")
    ax1.set_xlabel('Stock Price')
    ax1.set_ylabel('Time Till Expiration')
    ax1.set_zlabel(z_label)
    
    ax2 = plt.subplot(1, 2, 2, projection='3d')
    scatter_col(gt[0], gt[1], gt[2], 10, ax2)
    MSE = (L2(gt[-1].cpu(), model_pred[-1].cpu()).item())
    ax2.set_title("True output (MSE: %.12f)" % MSE)
    ax2.set_xlabel('Stock Price')
    ax2.set_ylabel('Time Till Expiration')
    ax2.set_zlabel(z_label)
    
    zlim = (min(ax1.get_zlim()[0], ax2.get_zlim()[0]),
            max(ax1.get_zlim()[1], ax2.get_zlim()[1]))
    ax1.set_zlim(zlim)
    ax2.set_zlim(zlim)
    
    plt.savefig(save_path)
    plt.close()


# Graph prediction
scatter_both([dataset[:,0].cpu(), dataset[:,1].cpu(), pred_value.detach().cpu().view(-1)],
             [dataset[:,0].cpu(), dataset[:,1].cpu(), dataset[:,-1].cpu()],
             'Options Price',
             'ModelOutput.png')

# Graph delta
pred_delta = model.get_delta(dataset[:,:2])
real_delta = bs.delta(dataset[:,0].cpu().numpy(), k.cpu().numpy(), 
                      dataset[:,1].cpu().numpy(), dataset[:,2].cpu().numpy(), dataset[:,3].cpu().numpy())
scatter_both([dataset[:,0].cpu(), dataset[:,1].cpu(), pred_delta.detach().cpu().view(-1)],
             [dataset[:,0].cpu(), dataset[:,1].cpu(), torch.as_tensor(real_delta)],
             'Delta',
             'Delta.png')

# Graph gamma
pred_gamma = model.get_gamma(dataset[:,:2])
real_gamma = bs.gamma(dataset[:,0].cpu().numpy(), k.cpu().numpy(), 
                      dataset[:,1].cpu().numpy(), dataset[:,2].cpu().numpy(), dataset[:,3].cpu().numpy())
scatter_both([dataset[:,0].cpu(), dataset[:,1].cpu(), pred_gamma.detach().cpu().view(-1)],
             [dataset[:,0].cpu(), dataset[:,1].cpu(), torch.as_tensor(real_gamma)],
             'Gamma',
             'Gamma.png')

# Graph theta
pred_theta = model.get_theta(dataset[:,:2])
real_theta = bs.theta(dataset[:,0].cpu().numpy(), k.cpu().numpy(),
                      dataset[:,1].cpu().numpy(), dataset[:,2].cpu().numpy(), dataset[:,3].cpu().numpy())
scatter_both([dataset[:,0].cpu(), dataset[:,1].cpu(), pred_theta.detach().cpu().view(-1)],
             [dataset[:,0].cpu(), dataset[:,1].cpu(), torch.as_tensor(real_theta)],
             'Theta',
             'Theta.png')

