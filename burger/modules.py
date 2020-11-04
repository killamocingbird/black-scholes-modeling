import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Feed forward neural network using only dense connections
class FCBlock(nn.Module):
    def __init__(self, neuron_list):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(neuron_list[i], neuron_list[i+1]) 
                                     for i in range(len(neuron_list)-1)])
    
    def forward(self, x, activation=F.relu):
        interm_out = x
        for layer in self.layers:
            interm_out = activation(layer(interm_out))
        return interm_out

# Linear layer with masked connectivity
class MaskedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, mask):
        """
       :param in_features: number of input features
       :param out_features: number of output features
       :param indices_mask: list of two lists containing indices for dimensions 0 and 1, used to create the mask
       """
        super(MaskedLinear, self).__init__()
 
        def backward_hook(grad):
            # Clone due to not being allowed to modify in-place gradients
            out = grad.clone()
            out[self.mask] = 0
            return out
 
        self.linear = nn.Linear(in_dim, out_dim)
        self.mask = mask
        self.linear.weight.data[self.mask] = 0 # zero out bad weights
        self.linear.weight.register_hook(backward_hook) # hook to zero out bad gradients
 
    def forward(self, x):
        return self.linear(x)

# Linear layer with branched connectivity
class BranchedLinear(nn.Module):
    def __init__(self, num_branches, in_features, out_features):
        super().__init__()
        self.num_branches = num_branches
        self.in_features = in_features
        self.out_features = out_features
        
        self.branches = nn.ModuleList([
            nn.Linear(in_features, out_features) for i in range(num_branches)
            ])
        
    def forward(self, x):
        assert x.shape[1] == self.num_branches * self.in_features, \
            "Invalid input shape, expected %d got %d" % (self.num_branches * self.in_features, x.shape[1])
        out = torch.zeros(x.shape[0], self.num_branches * self.out_features).to(x.device)
        for branch in range(self.num_branches):
            out[:,branch*self.out_features:(branch+1)*self.out_features] = \
                self.branches[branch](x[:,branch*self.in_features:(branch+1)*self.in_features])
        return out
            
        

# Conv Block featuring Convolution + Batchnorm + Pool + Activation with an
# option for coordinate injection
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_batch_norm=True, downsample=2, coord_conv=False):
        super().__init__()
        if coord_conv:
            self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, padding=padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.coord_conv = coord_conv
        self.batchnorm = None
        self.down = None
        
        if use_batch_norm:
            self.batchnorm = nn.BatchNorm2d(out_channels)
            
        if downsample > 1:
            self.down = nn.MaxPool2d(downsample, stride=downsample)
        
    def forward(self, x, return_input=False, activation=F.relu):
        if self.coord_conv:
            # Generate coordinate layers
            x_coords = torch.arange(x.shape[2]).expand(x.shape[3], x.shape[2]).float().t() / x.shape[2]
            y_coords = torch.arange(x.shape[3]).expand(x.shape[2], x.shape[3]).float() / x.shape[3]
            coords = torch.stack((x_coords, y_coords))
            coords = coords.expand(x.shape[0], *coords.shape).to(x.device)
            x = torch.cat((x, coords), 1)
        interm_out = activation(self.conv(x))
        if self.batchnorm is not None:
            interm_out = self.batchnorm(interm_out)
        if self.down is not None:
            interm_out = self.down(interm_out)
        
        if return_input:
            return x, interm_out
        else:
            return interm_out
        

# Foundation to base all models on
class Foundation(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []
    
    # Tracking losses
    def update_loss(self, train_loss, val_loss=None):
        self.train_loss.append(train_loss)
        if val_loss is not None:
            self.val_loss.append(val_loss)
            
    # Plots saved training curves
    def gen_train_plots(self, save_path='', header=''):
        if len(self.train_loss) < 1:
            raise "No losses to plot"
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title('Training Curve')
        plt.plot([i for i in range(len(self.train_loss))], self.train_loss, linestyle='dashed',
                 label='Training')
        if len(self.val_loss) >= 1:
            plt.plot([i for i in range(len(self.val_loss))], self.val_loss,
                     label='Validation')
        plt.legend()
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        
        plt.savefig(os.path.join(save_path, header + 'train_curve.png'))
        plt.close()
    
    # Predict a whole dataset on a specified device
    def predict_data(self, x, aux=None, f=None, batch_size=-1, device='cpu'):
        precision = x.type()
        with torch.no_grad():
            if batch_size == -1:
                if aux is not None:
                    if f is not None:
                        return self(x.type(precision).to(device), aux.type(precision).to(device), f)
                    else:
                        return self(x.type(precision).to(device), aux.type(precision).to(device))
                else:
                    return self(x.type(precision).to(device))
            else:
                # Get shape of output
                if aux is not None:
                    if f is not None:
                        out_shape = self(x[:2].type(precision).to(device),
                                         aux[:2].type(precision).to(device), f).shape
                    else:
                        out_shape = self(x[:2].type(precision).to(device),
                                         aux[:2].type(precision).to(device)).shape
                else:
                    out_shape = self(x[:2].type(precision).to(device)).shape
                out = torch.zeros(len(x), *out_shape[1:])
                for batch_idx in range(len(x) // batch_size + 1):
                    if batch_idx != (len(x) // batch_size) - 1:
                        xbatch = x[batch_idx*batch_size:(batch_idx+1)*batch_size].type(precision).to(device)
                        if aux is not None:
                            auxbatch = aux[batch_idx*batch_size:(batch_idx+1)*batch_size].type(precision).to(device)
                            if f is not None:
                                out[batch_idx*batch_size:(batch_idx+1)*batch_size] = self(xbatch, auxbatch, f).cpu()
                            else:
                                out[batch_idx*batch_size:(batch_idx+1)*batch_size] = self(xbatch, auxbatch).cpu()
                        else:
                            out[batch_idx*batch_size:(batch_idx+1)*batch_size] = self(xbatch).cpu()
                    else:
                        xbatch = x[batch_idx*batch_size:].type(precision).to(device)
                        if aux is not None:
                            auxbatch = aux[batch_idx*batch_size:].type(precision).to(device)
                            if f is not None:
                                out[batch_idx*batch_size:] = self(xbatch, auxbatch, f).cpu()
                            else:
                                out[batch_idx*batch_size:] = self(xbatch, auxbatch).cpu()
                        else:
                            out[batch_idx*batch_size:] = self(xbatch).cpu()
                return out
                
    
    def save(self, save_path='', header='', optimizer=None):
        checkpoint = {
            'state_dict': self.state_dict(),
            'train_loss': self.train_loss,
            'val_loss': self.val_loss
        }
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        torch.save(checkpoint, save_path + header + 'checkpoint.pth')
        torch.save(self, save_path + header + 'model.pth')
    
    def load(self, checkpoint):
        self.load_state_dict(checkpoint['state_dict'])
        self.train_loss = checkpoint['train_loss']
        self.val_loss = checkpoint['val_loss']
        
            
            
            
            
            
            
            
            
            