# -*- coding: utf-8 -*-
"""
@author: Huipeng Gu, Pengxiang Hong.
The benchmark test is testing architecture of FS-PINNs.
"""

"""
1. Packages imported
"""

import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from random import uniform
from torch.autograd import Variable

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda:0" if  torch.cuda.is_available() else "cpu")

"""
2. Networks design and parameters 
"""

global E, nu, alpha, c0, K, lambd, mu, betaFS
E       = 1.0
nu      = 0.3
alpha   = 1.0
c0      = 1.0
K       = 1.0
lambd   = E*nu / ((1+nu)*(1-2*nu))
mu      = E / (2*(1+nu))
betaFS  = 0.5*alpha*alpha/(lambd + mu)
T_final = 0.5

neurons_u = 0
neurons_p = 0

class Netu(nn.Module):
    def __init__(self):
        super(Netu, self).__init__()
        self.hidden_layer1 = nn.Linear(3,neurons_u)
        self.hidden_layer2 = nn.Linear(neurons_u,neurons_u)
        self.hidden_layer3 = nn.Linear(neurons_u,neurons_u)
        self.output_layer = nn.Linear(neurons_u,2)
        
    def forward(self, x):
        x = torch.tanh(self.hidden_layer1(x))
        x = torch.tanh(self.hidden_layer2(x))
        x = torch.tanh(self.hidden_layer3(x))
        x = self.output_layer(x)
        return x

class Netp(nn.Module):
    def __init__(self):
        super(Netp, self).__init__()
        self.hidden_layer1 = nn.Linear(3,neurons_p)
        self.hidden_layer2 = nn.Linear(neurons_p,neurons_p)
        self.hidden_layer3 = nn.Linear(neurons_p,neurons_p)
        self.output_layer = nn.Linear(neurons_p,1)
        
    def forward(self, x):
        x = torch.tanh(self.hidden_layer1(x))
        x = torch.tanh(self.hidden_layer2(x))
        x = torch.tanh(self.hidden_layer3(x))
        x = self.output_layer(x)
        return x
        

"""
3. Loss functions
"""

# PDE loss function for p 
def loss_Qs(netp, netu_old, netp_old, x, y, t):
    # the source or sink term Qs is defined here
    Qs = (c0 + 2*K)*torch.sin(x+y)*torch.exp(t) + alpha*(x+y)
    
    output_netp     = netp(torch.hstack( (x, y, t) ))
    output_netu_old = netu_old(torch.hstack( (x, y, t) ))
    output_netp_old = netp_old(torch.hstack( (x, y, t) ))
    
    p               = output_netp[:, 0].reshape(-1,1)
    p_old           = output_netp_old[:, 0].reshape(-1,1)
    u1_old          = output_netu_old[:, 0].reshape(-1,1)
    u2_old          = output_netu_old[:, 1].reshape(-1,1)
    
    p_t             = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_x             = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y             = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_xx            = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), retain_graph=True, create_graph=True)[0]
    p_yy            = torch.autograd.grad(p_y, y, grad_outputs=torch.ones_like(p_y), retain_graph=True, create_graph=True)[0]

    p_old_t         = torch.autograd.grad(p_old, t, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    
    u1_old_x        = torch.autograd.grad(u1_old, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    u2_old_y        = torch.autograd.grad(u2_old, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    u1_old_xt       = torch.autograd.grad(u1_old_x, t, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    u2_old_yt       = torch.autograd.grad(u2_old_y, t, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]    

    ret0 = (c0 + betaFS)*p_t - K*(p_xx + p_yy) + alpha*(u1_old_xt + u2_old_yt) - betaFS*p_old_t - Qs

    return ret0

# PDE loss function for u
def loss_f(netu, netp, x, y, t):
    # the body force term f is defined here
    f1 = -(lambd+2*mu)*t + alpha*torch.cos(x+y)*torch.exp(t)
    f2 = -(lambd+2*mu)*t + alpha*torch.cos(x+y)*torch.exp(t)
    
    output_netp = netp(torch.hstack( (x, y, t) ))
    output_netu = netu(torch.hstack( (x, y, t) ))

    p           = output_netp[:, 0].reshape(-1,1)
    u1          = output_netu[:, 0].reshape(-1,1)
    u2          = output_netu[:, 1].reshape(-1,1)

    p_x         = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y         = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

    u1_x        = torch.autograd.grad(u1, x, grad_outputs=torch.ones_like(u1), retain_graph=True, create_graph=True)[0]
    u1_y        = torch.autograd.grad(u1, y, grad_outputs=torch.ones_like(u1), retain_graph=True, create_graph=True)[0]
    u1_xx       = torch.autograd.grad(u1_x, x, grad_outputs=torch.ones_like(u1_x), retain_graph=True, create_graph=True)[0]
    u1_xy       = torch.autograd.grad(u1_x, y, grad_outputs=torch.ones_like(u1_x), retain_graph=True, create_graph=True)[0]
    u1_yx       = torch.autograd.grad(u1_y, x, grad_outputs=torch.ones_like(u1_y), retain_graph=True, create_graph=True)[0]
    u1_yy       = torch.autograd.grad(u1_y, y, grad_outputs=torch.ones_like(u1_y), retain_graph=True, create_graph=True)[0]

    u2_x        = torch.autograd.grad(u2, x, grad_outputs=torch.ones_like(u2), retain_graph=True, create_graph=True)[0]
    u2_y        = torch.autograd.grad(u2, y, grad_outputs=torch.ones_like(u2), retain_graph=True, create_graph=True)[0]
    u2_xx       = torch.autograd.grad(u2_x, x, grad_outputs=torch.ones_like(u2_x), retain_graph=True, create_graph=True)[0]
    u2_xy       = torch.autograd.grad(u2_x, y, grad_outputs=torch.ones_like(u2_x), retain_graph=True, create_graph=True)[0]
    u2_yx       = torch.autograd.grad(u2_y, x, grad_outputs=torch.ones_like(u2_y), retain_graph=True, create_graph=True)[0]
    u2_yy       = torch.autograd.grad(u2_y, y, grad_outputs=torch.ones_like(u2_y), retain_graph=True, create_graph=True)[0]
   
    ret1 = - mu*(2*u1_xx + u1_yy + u2_xy) - lambd*(u1_xx+u2_yx) + alpha*p_x - f1
    ret2 = - mu*(2*u2_yy + u2_xx + u1_yx) - lambd*(u1_xy+u2_yy) + alpha*p_y - f2
    
    return torch.hstack((ret1, ret2)) 

# PDE loss function for N
def loss_N(netn, x, y, t):
    # the body force term f, and the source or sink term f are defined here
    f1 = -(lambd+2*mu)*t + alpha*torch.cos(x+y)*torch.exp(t)
    f2 = -(lambd+2*mu)*t + alpha*torch.cos(x+y)*torch.exp(t)
    Qs = (c0 + 2*K)*torch.sin(x+y)*torch.exp(t) + alpha*(x+y)
    
    output_netn = netn(torch.hstack( (x, y, t) ))
    
    u1          = output_netn[:, 0].reshape(-1,1)
    u2          = output_netn[:, 1].reshape(-1,1)
    p           = output_netn[:, 2].reshape(-1,1)

    u1_x        = torch.autograd.grad(u1, x, grad_outputs=torch.ones_like(u1), retain_graph=True, create_graph=True)[0]
    u1_y        = torch.autograd.grad(u1, y, grad_outputs=torch.ones_like(u1), retain_graph=True, create_graph=True)[0]
    u1_xx       = torch.autograd.grad(u1_x, x, grad_outputs=torch.ones_like(u1_x), retain_graph=True, create_graph=True)[0]
    u1_xy       = torch.autograd.grad(u1_x, y, grad_outputs=torch.ones_like(u1_x), retain_graph=True, create_graph=True)[0]
    u1_yx       = torch.autograd.grad(u1_y, x, grad_outputs=torch.ones_like(u1_y), retain_graph=True, create_graph=True)[0]
    u1_yy       = torch.autograd.grad(u1_y, y, grad_outputs=torch.ones_like(u1_y), retain_graph=True, create_graph=True)[0]
    u1_xt       = torch.autograd.grad(u1_x, t, grad_outputs=torch.ones_like(u1_x), retain_graph=True, create_graph=True)[0]

    u2_x        = torch.autograd.grad(u2, x, grad_outputs=torch.ones_like(u2), retain_graph=True, create_graph=True)[0]
    u2_y        = torch.autograd.grad(u2, y, grad_outputs=torch.ones_like(u2), retain_graph=True, create_graph=True)[0]
    u2_xx       = torch.autograd.grad(u2_x, x, grad_outputs=torch.ones_like(u2_x), retain_graph=True, create_graph=True)[0]
    u2_xy       = torch.autograd.grad(u2_x, y, grad_outputs=torch.ones_like(u2_x), retain_graph=True, create_graph=True)[0]
    u2_yx       = torch.autograd.grad(u2_y, x, grad_outputs=torch.ones_like(u2_y), retain_graph=True, create_graph=True)[0]
    u2_yy       = torch.autograd.grad(u2_y, y, grad_outputs=torch.ones_like(u2_y), retain_graph=True, create_graph=True)[0]
    u2_yt       = torch.autograd.grad(u2_y, t, grad_outputs=torch.ones_like(u2_y), retain_graph=True, create_graph=True)[0]
    
    p_t         = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_x         = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y         = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_xx        = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_yy        = torch.autograd.grad(p_y, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    
    ret1 = - mu*(2*u1_xx + u1_yy + u2_xy) - lambd*(u1_xx+u2_yx) + alpha*p_x - f1
    ret2 = - mu*(2*u2_yy + u2_xx + u1_yx) - lambd*(u1_xy+u2_yy) + alpha*p_y - f2
    ret3 = c0*p_t - K*(p_xx + p_yy) + alpha*(u1_xt + u2_yt) - Qs
    
    return torch.hstack((ret1, ret2, ret3))
    
"""
4. Training model 
"""
# Data points
N_point  = 10000 
N_right  = 200
N_left   = 200
N_upper  = 200
N_lower  = 200
N_bottom = 500

# iteration setting
out_iter = 20
in_iter  = 3000

# recording list
L2_time1_epoch = []
L2_p = []
L2_u1 = []
L2_u2 = []
Train_Time = []

# neuron list of two networks
neuron_list = [ [5,25],[10,20],[15,15],[20,10],[25,5]  ]

# learning rate setting
decreasing_FS = 1
LR_FS = 0.0001

# loss function
Loss = torch.nn.MSELoss()

# different combo of FS-PINN network architecture
for neurons_u, neurons_p in neuron_list:
    
    netu     = Netu()
    netu_old = Netu()
    netp     = Netp()
    netp_old = Netp()
    netu     = netu.to(device)
    netu_old = netu_old.to(device)
    netp     = netp.to(device)        
    netp_old = netp_old.to(device) 
    print("\nCase:N_p={0} and N_u={1} with in_iter={2} and out_iter={3}".format(neurons_p, neurons_u, in_iter, out_iter))

    Time1_Sum = 0
    Time2_Sum = 0
    Time1_l2_list = []
    p_l2_list = []
    u1_l2_list = []
    u2_l2_list  = []
    
    """
    network netu/netp initial and netu_old/netp_old zero initial
    """
    for m in netu.modules():  
        if isinstance(m, torch.nn.Linear):  
            torch.nn.init.xavier_uniform_(m.weight) 
    for m in netp.modules(): 
        if isinstance(m, torch.nn.Linear): 
            torch.nn.init.xavier_uniform_(m.weight)
    for m in netu_old.modules():
        if isinstance(m, torch.nn.Linear):
            m.weight.data.fill_(0.0)
            m.bias.data.fill_(0.0)
    for m in netp_old.modules():
        if isinstance(m, torch.nn.Linear):
            m.weight.data.fill_(0.0)
            m.bias.data.fill_(0.0)

    # Data points used to calculate the L2-error
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    m,n = x.shape[0], y.shape[0]
    X, Y = np.meshgrid(x, y)
    point = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    t = np.ones((m * n, 1)) * T_final
    inputs = np.hstack((point, t))

    u1_ex = 0.5 * inputs[:, 2] * inputs[:, 0] * inputs[:, 0]
    u2_ex = 0.5 * inputs[:, 2] * inputs[:, 1] * inputs[:, 1]
    p_ex = np.sin(inputs[:, 0] + inputs[:, 1]) * np.exp(inputs[:, 2])

    u1_ex = Variable(torch.from_numpy(u1_ex.reshape(-1,1)).float(), requires_grad=False).to(device)
    u2_ex = Variable(torch.from_numpy(u2_ex.reshape(-1,1)).float(), requires_grad=False).to(device)
    p_ex = Variable(torch.from_numpy(p_ex.reshape(-1,1)).float(), requires_grad=False).to(device)
    inputs = Variable(torch.from_numpy(inputs).float(), requires_grad=False).to(device)

    # initial L2-error of FS-PINN
    out_p = netp(inputs).reshape(-1, 1)
    output = netu(inputs)
    out_u1 = output[:, 0].reshape(-1, 1)
    out_u2 = output[:, 1].reshape(-1, 1)
    p_abs_error = torch.abs(p_ex - out_p).reshape(-1, 1)
    u1_abs_error = torch.abs(u1_ex - out_u1).reshape(-1, 1)
    u2_abs_error = torch.abs(u2_ex - out_u2).reshape(-1, 1)
    p_l2 = torch.sum(p_abs_error ** 2 / (m * n))
    u1_l2 = torch.sum(u1_abs_error ** 2 / (m * n))
    u2_l2 = torch.sum(u2_abs_error ** 2 / (m * n))
    
    l2_FS = p_l2 + u1_l2 + u2_l2
    Time1_l2_list = [[0, 0, l2_FS.item()]]
    
    p_l2_list.append(p_l2.item())
    u1_l2_list.append(u1_l2.item())
    u2_l2_list.append(u2_l2.item())
    
    # training
    for out_epoch in range(out_iter):

        # Data
        # boundary and initial points
        Right  = np.hstack( (np.ones((N_right, 1), dtype=float), np.random.rand(N_right, 1), np.random.rand(N_right, 1)) )
        Left   = np.hstack( (np.zeros((N_left, 1), dtype=float), np.random.rand(N_right, 1), np.random.rand(N_right, 1)) )  
        Upper  = np.hstack( (np.random.rand(N_upper, 1), np.ones((N_upper, 1), dtype=float), np.random.rand(N_upper, 1)) )
        Lower  = np.hstack( (np.random.rand(N_lower, 1), np.zeros((N_lower, 1), dtype=float), np.random.rand(N_lower, 1)) )
        Bottom = np.hstack( (np.random.rand(N_bottom, 1), np.random.rand(N_bottom, 1), np.zeros((N_bottom, 1), dtype=float)) )
        X_u_train = np.vstack((Right, Left, Upper, Lower, Bottom))
    
        u1_train = 0.5 * X_u_train[:, 2] * X_u_train[:, 0] * X_u_train[:, 0]
        u2_train = 0.5 * X_u_train[:, 2] * X_u_train[:, 1] * X_u_train[:, 1]
        p_train = np.sin(X_u_train[:, 0] + X_u_train[:, 1]) * np.exp(X_u_train[:, 2])
    
        # collocation points
        x_collocation = np.random.rand(N_point, 1)
        y_collocation = np.random.rand(N_point, 1)
        t_collocation = np.random.rand(N_point, 1) * 1.1 * T_final
        pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
        pt_y_collocation = Variable(torch.from_numpy(y_collocation).float(), requires_grad=True).to(device)
        pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    
        optimizer_p = torch.optim.Adam(netp.parameters(), lr = LR_FS)
        scheduler_p = torch.optim.lr_scheduler.ExponentialLR(optimizer_p, decreasing_FS)
        start1 = time.perf_counter()
        for in_epoch in range(in_iter):
            # train netp using netp_old, netu_old
            optimizer_p.zero_grad()
    
            # boundary and initial conditions for netp
            torch_X_train = Variable(torch.from_numpy(X_u_train).float(), requires_grad=False).to(device)
            torch_u_train = Variable(torch.from_numpy(p_train.reshape(-1,1)).float(), requires_grad=False).to(device)
    
            net_bc_out1 = netp(torch_X_train)
            mse_p = Loss(net_bc_out1, torch_u_train)
            
            # PDE loss for netp
            all_zeros = np.zeros((N_point,1))
            pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
            
            pde_p = loss_Qs(netp, netu_old, netp_old, pt_x_collocation, pt_y_collocation, pt_t_collocation)
            mse_Qs = Loss(pde_p, pt_all_zeros)
            
            # total loss for step1
            loss1 = mse_p + mse_Qs
            loss1.backward()
            optimizer_p.step()
            scheduler_p.step()
            
            # L2-error of p
            out_p = netp(inputs).reshape(-1, 1)
            p_abs_error = torch.abs(p_ex - out_p).reshape(-1, 1)
            p_l2 = torch.sum(p_abs_error ** 2 / (m * n))
            
            p_l2_list.append(p_l2.item())

        end1 = time.perf_counter()
        optimizer_u = torch.optim.Adam(netu.parameters(), lr = LR_FS)
        scheduler_u = torch.optim.lr_scheduler.ExponentialLR(optimizer_u, decreasing_FS)
        start2 = time.perf_counter()  
        for in_epoch in range(in_iter):
            # train netu using netp
            optimizer_u.zero_grad()
    
            # boundary and initial conditions for netu
            u_train  = np.hstack( (u1_train.reshape(-1, 1), u2_train.reshape(-1, 1)) )
            torch_X_train = Variable(torch.from_numpy(X_u_train).float(), requires_grad=False).to(device)
            torch_u_train = Variable(torch.from_numpy(u_train).float(), requires_grad=False).to(device)
    
            net_bc_out2 = netu(torch_X_train)
            mse_u = Loss(net_bc_out2, torch_u_train)
            
            # PDE loss for netu
            all_zeros = np.zeros((N_point,2))
            pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
            
            pde_u = loss_f(netu, netp, pt_x_collocation, pt_y_collocation, pt_t_collocation)
            mse_f = Loss(pde_u, pt_all_zeros)
            
            # total loss for step2
            loss2 = mse_u + mse_f
            loss2.backward()
            optimizer_u.step()
            scheduler_u.step()
            
            # L2 -error of u
            output = netu(inputs)
            out_u1 = output[:,0].reshape(-1, 1)
            out_u2 = output[:,1].reshape(-1, 1)
            u1_abs_error = torch.abs(u1_ex - out_u1).reshape(-1, 1)
            u2_abs_error = torch.abs(u2_ex - out_u2).reshape(-1, 1)
            u1_l2 = torch.sum(u1_abs_error ** 2 / (m * n))
            u2_l2 = torch.sum(u2_abs_error ** 2 / (m * n))
            
            u1_l2_list.append(u1_l2.item())
            u2_l2_list.append(u2_l2.item())
    
        end2 = time.perf_counter()
    
        # L2-error of p and u  at every iteration
        out_p = netp(inputs).reshape(-1, 1)
        output = netu(inputs)
        out_u1 = output[:,0].reshape(-1, 1)
        out_u2 = output[:,1].reshape(-1, 1)
        p_abs_error = torch.abs(p_ex - out_p).reshape(-1, 1)
        u1_abs_error = torch.abs(u1_ex - out_u1).reshape(-1, 1)
        u2_abs_error = torch.abs(u2_ex - out_u2).reshape(-1, 1)
        p_l2 = torch.sum(p_abs_error ** 2 / (m * n))
        u1_l2 = torch.sum(u1_abs_error ** 2 / (m * n))
        u2_l2 = torch.sum(u2_abs_error ** 2 / (m * n))

        l2_FS = p_l2 + u1_l2 + u2_l2

        # updating netp_old and  netu_old
        netp_old = deepcopy(netp)
        netu_old = deepcopy(netu)

        # recording  training time
        Time1 = end1 - start1 
        Time2 = end2 - start2
        Time1_Sum = Time1_Sum + Time1
        Time2_Sum = Time2_Sum + Time2

        Time1_l2_list.append([Time1_Sum+Time2_Sum, (out_epoch+1)*in_iter, l2_FS.item()])
        print('iteration:',out_epoch+1," Loss1:",loss1.item()," Loss2:",loss2.item()," Time:",Time1+Time2)
    
    Time = Time1_Sum + Time2_Sum
    print("Total Training Time:", Time)
    Train_Time.append([Time1_Sum, Time2_Sum, Time, Time1_Sum/Time, Time2_Sum/Time ])
    Time1_l2_list = np.array(Time1_l2_list).reshape(-1,3)
    
    L2_time1_epoch.append(Time1_l2_list)
    L2_p.append(p_l2_list)
    L2_u1.append(u1_l2_list)
    L2_u2.append(u2_l2_list)


"""
5. Visualization
"""
# print the information of training time
print("\nTotal Train_Time: Time1,  Time2,  Time1+Time2,  Time1/(Time1+Time2),  Time2/(Time1+Time2)")
for lst in Train_Time:
    print(lst)

#  Epochs  vs  L2-error
# p
fig=plt.figure()
ax = fig.add_subplot(111)
for i in range(len(L2_p)):
    ax.semilogy(range(len(L2_p[i])),L2_p[i], linewidth = '0.7', alpha = 0.5)
plt.legend(['$N_u={0}$:$N_p={1}$'.format(neuron_list[i][0], neuron_list[i][1]) for i in range(len(L2_p))])
plt.xlabel('Epochs')
plt.ylabel('MSE for pressure $p$')

#  u1
fig=plt.figure()
ax = fig.add_subplot(111)
for i in range(len(L2_u1)):
    ax.semilogy(range(len(L2_u1[i])),L2_u1[i], linewidth = '0.7', alpha = 0.5)
plt.legend(['$N_u={0}$:$N_p={1}$'.format(neuron_list[i][0], neuron_list[i][1]) for i in range(len(L2_u1))])
plt.xlabel('Epochs')
plt.ylabel('MSE for x-displacement $u_1$')

# u2
fig=plt.figure()
ax = fig.add_subplot(111)
for i in range(len(L2_u2)):
    ax.semilogy(range(len(L2_u2[i])),L2_u2[i], linewidth = '0.7', alpha = 0.5)
plt.legend(['$N_u={0}$:$N_p={1}$'.format(neuron_list[i][0], neuron_list[i][1]) for i in range(len(L2_u2))])
plt.xlabel('Epochs')
plt.ylabel('MSE for y-displacement $u_2$')

# total
fig=plt.figure()
ax = fig.add_subplot(111)
for i in range(len(L2_p)):
    L2 = [x + y + z for x, y, z in zip(L2_p[i], L2_u1[i], L2_u2[i])]
    ax.semilogy(range(len(L2_p[i])),L2 , linewidth = '0.7', alpha = 0.5)
plt.legend(['$N_u={0}$:$N_p={1}$'.format(neuron_list[i][0], neuron_list[i][1]) for i in range(len(L2_p))])
plt.xlabel('Epochs')
plt.ylabel('Total MSE')

plt.show()
