import math
import numpy as np
import os
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F

import modules as m

L2 = nn.MSELoss()

class scaled_tanh(nn.Tanh):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.tanh(x)

class BS_PDE(m.Foundation):
    def __init__(self, device):
        super().__init__()
        # Input: (S, S_0, Tau)
        self.layers = nn.Sequential(
            nn.Linear(2, 20),
            #nn.Tanh(),
            scaled_tanh(),
            nn.Linear(20, 20),
            #nn.Tanh(),
            scaled_tanh(),
            nn.Linear(20, 20),
            #nn.Tanh(),
            scaled_tanh(),
            nn.Linear(20, 20),
            #nn.Tanh(),
            scaled_tanh(),
            nn.Linear(20, 20),
            #nn.Tanh(),
            scaled_tanh(),
            nn.Linear(20, 1)
        )
        self.sigma = nn.Parameter(torch.tensor(.5, requires_grad=True))
        self.device = device
        #self.sigma = torch.as_tensor(0.65)
    
    # x: [batch, 2]
    def forward(self, x):
        return self.layers(x)
    
    # Enforces the PDE's structure
    # x: [batch, 2] (S, Tau)
    # sigma: [batch]
    # r: [batch]
    def pde_structure(self, x, r, return_forward=True):
        sigma = self.sigma
        x.requires_grad = True
        V = self.forward(x)
        
        Jacocbian_1, Jacobian_2 = second_order_jacobians(V, x)
        
        dVdt =  Jacocbian_1[:,1,0]
        ddVdSS = Jacobian_2[:,0,0]
        dVdS = Jacocbian_1[:,0,0]
        
        # Might break
        h = -dVdt + .5 * sigma**2 * x[:,0]**2 * ddVdSS + r * x[:,0] * dVdS - r * V.squeeze()
        
        if return_forward:
            return h, V
        else:
            return h
    
    """
    Functions to get greeks using finite-difference method
    """
    def get_delta(self, x):
        x.requires_grad = True
        V = self.forward(x)
        
        Gradient, _ = second_order_jacobians(V, x)
        return Gradient[:,0,0]
    
    def get_gamma(self, x):
        x.requires_grad = True
        V = self.forward(x)
        
        _, Jacobian_2 = second_order_jacobians(V, x)
        return Jacobian_2[:,0,0]
    
    def get_theta(self, x):
        x.requires_grad = True
        V = self.forward(x)
        
        Gradient, _ = second_order_jacobians(V, x)
        return -Gradient[:,1,0]
    
    
    # Enforces boundary conditions
    def boundary_conditions(self, x, k, r, epoch=0):
        # X = [S, T]
        
        # Create boundary conditions
        # V(0, Tau) = 0
        B_1 = x.clone()
        B_1[:,0] = 0
        B_1 = torch.unique(B_1, dim=0)
        # V(S, 0) = max(S - K, 0)
        B_2 = x.clone()
        B_2[:,1] = 0
        B_2 = torch.unique(B_2, dim=0)
        # V(S, inf) = S
        B_3 = x.clone()
        B_3[B_3[:,0] < 2.1,0] = 2.1
        B_3 = torch.unique(B_3, dim=0)
        
        # Pulse function setup
        #delta = self.get_delta(B_3)
        #pulsed_delta = torch.sin(math.pi * (.1* delta - 1)) / (10 * (delta - 1))
        
        b_1 = self.forward(B_1)
        b_2 = self.forward(B_2)
        b_3 = self.forward(B_3)
        
        return L2(b_1, torch.zeros(b_1.shape).to(b_1.device)) + \
               L2(b_2.squeeze(), F.relu(B_2[:,0] - k))# + \
               #L2(b_3.squeeze(), B_3[:,0] - k * torch.exp(-r * B_3[:,1]))
               #(pulsed_delta * (b_3.squeeze() - B_3[:,0])**2).mean()
    

class BS_SDE(BS_PDE):
    def __init__(self, device):
        super().__init__(device)
        
    def sde_conditions(self, x, x_prev, r):
        # x: [S, T], x_prev: [S, T]
        f = self.forward(x).squeeze()
        
        sigma = self.sigma
        x_prev.requires_grad = True
        f_prev = self.forward(x_prev)
        
        # Calculate partial for prev
        Jacocbian_1, Jacobian_2 = second_order_jacobians(f_prev, x_prev)
        
        dfdt =  Jacocbian_1[:,1,0]
        ddfdSS = Jacobian_2[:,0,0]
        dfdS = Jacocbian_1[:,0,0]
        
        # Calculate delta t
        delta_t = x[:,1] - x_prev[:,1]
        
        # Sample from normal distribution
        Z = torch.normal(torch.zeros(len(delta_t)), torch.ones(len(delta_t))).to(self.device)
        
        u = f_prev.squeeze() + delta_t * (dfdt + r*x_prev[:,0]*dfdS + 
                                .5*(x_prev[:,0] * sigma)**2 * ddfdSS) + \
            sigma * x_prev[:,0] * f_prev.squeeze() * (delta_t)**(.5) * Z
        
        return L2(u, f)
        
    
        
class Discrete_BS_PDE(m.Foundation):
    def __init__(self, delta_t, q, device):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, q)
        )
        self.sigma = nn.Parameter(torch.tensor(.5, requires_grad=True))
        
        # Scalar
        self.delta_t = delta_t
        
        # Load in Runge Kutta weights
        self.load_runge_kutta_weights(q, device=device)
    
    def load_runge_kutta_weights(self, q, path='IRK_weights', device='cpu'):
        target_file = os.path.join(path, 'Butcher_IRK%d.txt' % (q))
        
        # Check file path exists
        assert os.path.isfile(target_file), "Runge Kutta weights could not be found"
        
        # Load in weights
        tmp = np.float32(np.loadtxt(target_file, ndmin = 2))
        weights =  np.reshape(tmp[0:q**2+q], (q+1,q))    
        self.IRK_alpha = torch.as_tensor(weights[0:-1,:]).to(device)
        self.IRK_beta = torch.as_tensor(weights[-1:,:]).to(device)     
        self.IRK_times = torch.as_tensor(tmp[q**2+q:]).to(device)
    
    # Returns forward of network
    # [batch, q]
    def forward(self, x):
        return self.layers(x)
    
    # Returns [N[u^n, lambda], i], u (if return_forward) where N[u^n, lambda], i = N[u^(n+c_i), lambda]
    # [batch, q]
    # Assumes working with T (dT/dt = -1)
    # x = [S, Tau]
    def pde_structure(self, x, r, return_forward=True):
        sigma = self.sigma
        x.requires_grad = True
        V = self.forward(x)
        
        Jacobian_1, Jacobian_2 = second_order_jacobians(V, x)
        dVdt =  Jacobian_1[:,1,:]
        ddVdSS = Jacobian_2[:,0,:]
        dVdS = Jacobian_1[:,0,:]
        
        # Might break
        h = -dVdt + (ddVdSS.t() * .5 * sigma**2 * x[:,0]**2).t() + r * (dVdS.t() * x[:,0]).t() - r * V.squeeze()
        
        if return_forward:
            return h, V
        else:
            return h
    
    # Returns u^n = u^(n+c_i) + delta_t * dot(alphai, N[u^(n+c_i), lambda])
    def get_u_0(self, x, r):
        h, V = self.pde_structure(x, r)
        u_0 = V + self.delta_t * h @ self.IRK_alpha.t()
        return u_0
    
    # Returns u^(n+1) = u^(n+c_i) + delta_t * dot(alphai - beta, N[u^(n+c_i), lambda])
    def get_u_1(self, x, r):
        h, V = self.pde_structure(x, r)
        u_1 = V + self.delta_t * h @ (self.IRK_alpha - self.IRK_beta).t()
        return u_1        


class Burgers_PDE(m.Foundation):
    def __init__(self, device):
        super().__init__()
        # Input: (S, S_0, Tau)
        self.layers = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
        self.lambda_1 = nn.Parameter(torch.tensor(0., requires_grad=True))
        self.lambda_2 = nn.Parameter(torch.tensor(-6., requires_grad=True))
    
    def forward(self, x):
        return self.layers(x)
    
    def pde_structure(self, x, return_forward=True):
        lambda_1 = self.lambda_1        
        lambda_2 = torch.exp(self.lambda_2)
        x.requires_grad = True
        u = self.forward(x)
        Jacobian_1, Jacobian_2 = second_order_jacobians(u, x)
        
        u_x = Jacobian_1[:,0,0]
        u_t = Jacobian_1[:,1,0]
        u_xx = Jacobian_2[:,0,0]
        
        h = u_t + lambda_1*u*u_x - lambda_2*u_xx

        if return_forward:
            return h, u
        else:
            return h


class Discrete_Burgers_PDE(m.Foundation):
    def __init__(self, delta_t, q, device, lb=-1, ub=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, q),
        )
        self.lambda_1 = nn.Parameter(torch.tensor(0., requires_grad=True))
        self.lambda_2 = nn.Parameter(torch.tensor(-6., requires_grad=True))
        
        # Scalar
        self.delta_t = delta_t
        
        # Load in Runge Kutta weights
        self.load_runge_kutta_weights(max(q, 1), device=device)
        
        self.lb = lb
        self.ub = ub
    
    def load_runge_kutta_weights(self, q, path='IRK_weights', device='cpu'):
        target_file = os.path.join(path, 'Butcher_IRK%d.txt' % (q))
        
        # Check file path exists
        assert os.path.isfile(target_file), "Runge Kutta weights could not be found"
        
        # Load in weights
        tmp = np.float32(np.loadtxt(target_file, ndmin = 2))
        weights =  np.reshape(tmp[0:q**2+q], (q+1,q))    
        self.IRK_alpha = torch.as_tensor(weights[0:-1,:]).to(device)
        self.IRK_beta = torch.as_tensor(weights[-1:,:]).to(device)     
        self.IRK_times = torch.as_tensor(tmp[q**2+q:]).to(device)
    
    # Returns forward of network
    # [batch, q]
    def forward(self, x):
        # Normalize
        x_star = 2.0*(x - self.lb)/(self.ub - self.lb) - 1.0
        return self.layers(x_star)
    
    # Returns [N[u^n, lambda], i], u (if return_forward) where N[u^n, lambda], i = N[u^(n+c_i), lambda]
    # [batch, q]
    def pde_structure(self, x, return_forward=True):
        lambda_1 = self.lambda_1        
        lambda_2 = torch.exp(self.lambda_2)
        x.requires_grad = True
        u = self.forward(x)
        Jacobian_1, Jacobian_2 = second_order_jacobians(u, x)
        
        u_x = Jacobian_1[:,0,:]
        u_xx = Jacobian_2[:,0,:]
        
        h = lambda_1*u*u_x - lambda_2*u_xx

        if return_forward:
            return h, u
        else:
            return h
    
    # Returns u^n = u^(n+c_i) + delta_t * dot(alphai, N[u^(n+c_i), lambda])
    def get_u_0(self, x):
        h, u = self.pde_structure(x)
        u_0 = u + self.delta_t * h @ self.IRK_alpha.t()
        return u_0
    
    # Returns u^(n+1) = u^(n+c_i) + delta_t * dot(alphai - beta, N[u^(n+c_i), lambda])
    def get_u_1(self, x):
        h, u = self.pde_structure(x)
        u_1 = u + self.delta_t * h @ (self.IRK_alpha - self.IRK_beta).t()
        return u_1
    
    

def nth_derivative(f, wrt, n, create_graph=True):
    for i in range(n):
        grads = grad(f, wrt, create_graph=create_graph, grad_outputs=torch.ones(f.shape).to(f.device))[0]
        f = grads
    return grads

# Creates second order partials for a scalar function f
def second_order_partials(f, wrt, create_graph=True):
    gradient = grad(f, wrt, create_graph=create_graph, grad_outputs=torch.ones(f.shape).to(f.device))[0]
    hessian = grad(gradient, wrt, create_graph=create_graph, grad_outputs=torch.ones(gradient.shape).to(gradient.device))[0]
    
    return gradient, hessian

# Creates first and second order jacobian for vector function f
def second_order_jacobians(f, wrt, create_graph=True):
    jacobian_1 = []
    jacobian_2 = []
    for i in range(f.shape[1]):
        j1 = grad(f[:,i:i+1], wrt, create_graph=create_graph, grad_outputs=torch.ones(f[:,i:i+1].shape).to(f.device))[0]
        j2 = grad(j1, wrt, create_graph=create_graph, grad_outputs=torch.ones(j1.shape).to(j1.device))[0]
        jacobian_1.append(j1)
        jacobian_2.append(j2)
    jacobian_1 = torch.stack(jacobian_1, -1)
    jacobian_2 = torch.stack(jacobian_2, -1)
    return jacobian_1, jacobian_2
        
        
    
    
    
    
    
    


