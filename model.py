import math
import numpy as np
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F

import modules as m

L2 = nn.MSELoss()

class BS_PDE(m.Foundation):
    def __init__(self, device):
        super().__init__()
        # Input: (S, S_0, Tau)
        self.layers = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.sigma = nn.Parameter(torch.tensor(0.5, requires_grad=True))
    
    # x: [batch, 3]
    def forward(self, x):
        return self.layers(x)
    
    # Enforces the PDE's structure
    # x: [batch, 3]
    # sigma: [batch]
    # r: [batch]
    def pde_structure(self, x, r, return_forward=True):
        x.requires_grad = True
        V = self.forward(x)
        
        Gradient, Hessian = second_order_partials(V, x)
        dVdt =  Gradient[:,2]
        ddVdSS = Hessian[:,0]
        dVdS = Gradient[:,0]
        
        # Might break
        h = -dVdt + .5 * (self.sigma * x[:,0])**2 * ddVdSS + r * x[:,0] * dVdS - r * V.squeeze()
        
        if return_forward:
            return h, V
        else:
            return h
    
    """
    Functions to get greeks
    """
    def get_delta(self, x):
        x.requires_grad = True
        V = self.forward(x)
        
        Gradient, _ = second_order_partials(V, x)
        return Gradient[:,0]
    
    def get_gamma(self, x):
        x.requires_grad = True
        V = self.forward(x)
        
        _, Hessian = second_order_partials(V, x)
        return Hessian[:,0]
    
    def get_theta(self, x):
        x.requires_grad = True
        V = self.forward(x)
        
        Gradient, _ = second_order_partials(V, x)
        return -Gradient[:,2]
    
    
    # Enforces boundary conditions
    def boundary_conditions(self, x, r, epoch=0):
        # X = [S, K, T]
        
        # Create boundary conditions
        # V(0, Tau) = 0
        B_1 = x.clone()
        B_1[:,0] = 0
        # V(S, 0) = max(S - K, 0)
        B_2 = x.clone()
        B_2[:,2] = 0
        # V(S, inf) = S
        B_3 = x.clone()
        B_3[B_3[:,0] < 2,0] = 2
        
        # Pulse function setup
        #delta = self.get_delta(B_3)
        #pulsed_delta = torch.sin(math.pi * (.1* delta - 1)) / (10 * (delta - 1))
        
        b_1 = self.forward(B_1)
        b_2 = self.forward(B_2)
        b_3 = self.forward(B_3)
        
        
        return L2(b_1, torch.zeros(b_1.shape).to(b_1.device)) + \
               L2(b_2.squeeze(), F.relu(B_2[:,0] - B_2[:,1])) + \
               L2(b_3.squeeze(), B_3[:,0] - B_3[:,1] * torch.exp(-r * B_3[:,2]))
               #(pulsed_delta * (b_3.squeeze() - B_3[:,0])**2).mean()
        
    
def nth_derivative(f, wrt, n, create_graph=True):
    for i in range(n):
        grads = grad(f, wrt, create_graph=create_graph, grad_outputs=torch.ones(f.shape).to(f.device))[0]
        f = grads
    return grads

def second_order_partials(f, wrt, create_graph=True):
    gradient = grad(f, wrt, create_graph=create_graph, grad_outputs=torch.ones(f.shape).to(f.device))[0]
    hessian = grad(gradient, wrt, create_graph=create_graph, grad_outputs=torch.ones(gradient.shape).to(gradient.device))[0]
    
    return gradient, hessian

    
    
    
    
    
    
    


