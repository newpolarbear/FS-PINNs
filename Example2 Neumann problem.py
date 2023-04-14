# -*- coding: utf-8 -*-
"""
@author: Huipeng Gu, Pengxiang Hong.
The benchmark test is a Neumann problem.

"""

"""
1. Packages imported
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
2. Networks design and parameters 
"""

global E, nu, alpha, c0, K, lambd, mu, betaFS
E = 1.0
nu = 0.3
alpha = 1.0
c0 = 1.0
K = 1.0
lambd = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
betaFS = 0.5 * alpha * alpha / (lambd + mu)
T_final = 0.2

neurons_u = 15
neurons_p = 15


class Netu(nn.Module):
    def __init__(self):
        super(Netu, self).__init__()
        self.hidden_layer1 = nn.Linear(3, neurons_u)
        self.hidden_layer2 = nn.Linear(neurons_u, neurons_u)
        self.hidden_layer3 = nn.Linear(neurons_u, neurons_u)
        self.output_layer = nn.Linear(neurons_u, 2)

    def forward(self, x):
        x = torch.tanh(self.hidden_layer1(x))
        x = torch.tanh(self.hidden_layer2(x))
        x = torch.tanh(self.hidden_layer3(x))
        x = self.output_layer(x)
        return x


class Netp(nn.Module):
    def __init__(self):
        super(Netp, self).__init__()
        self.hidden_layer1 = nn.Linear(3, neurons_p)
        self.hidden_layer2 = nn.Linear(neurons_p, neurons_p)
        self.hidden_layer3 = nn.Linear(neurons_p, neurons_p)
        self.output_layer = nn.Linear(neurons_p, 1)

    def forward(self, x):
        x = torch.tanh(self.hidden_layer1(x))
        x = torch.tanh(self.hidden_layer2(x))
        x = torch.tanh(self.hidden_layer3(x))
        x = self.output_layer(x)
        return x


"""
3. Functions
"""


# exact solution
def sol_u1(x, y, t):
    return torch.exp(-t) * (torch.sin(2 * torch.pi * y) * (torch.cos(2 * torch.pi * x) - 1) +
                            torch.sin(torch.pi * x) * torch.sin(torch.pi * x) / (mu + lambd))


def sol_u2(x, y, t):
    return torch.exp(-t) * (torch.sin(2 * torch.pi * x) * (1 - torch.cos(2 * torch.pi * y)) +
                            torch.sin(torch.pi * x) * torch.sin(torch.pi * x) / (mu + lambd))


def sol_p(x, y, t):
    return torch.exp(-t) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


# model function
def Qs(x, y, t):
    Qs = (-c0 + 2 * torch.pi * torch.pi * K) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.exp(
        -t) - alpha * torch.pi / (mu + lambd) * torch.sin(torch.pi * (x + y)) * torch.exp(-t)
    return Qs


def f1(x, y, t):
    f1 = (8 * mu * torch.pi * torch.pi * torch.sin(2 * torch.pi * y) * torch.cos(
        2 * torch.pi * x) - 4 * mu * torch.pi * torch.pi * torch.sin(2 * torch.pi * y)
          + torch.pi * torch.pi * (3 * mu + lambd) / (mu + lambd) * torch.sin(torch.pi * x) * torch.sin(
                torch.pi * y) - torch.pi * torch.pi * torch.cos(torch.pi * x) * torch.cos(torch.pi * y)
          + alpha * torch.pi * torch.cos(torch.pi * x) * torch.sin(torch.pi * y)) * torch.exp(-t)
    return f1


def f2(x, y, t):
    f2 = (-8 * mu * torch.pi * torch.pi * torch.sin(2 * torch.pi * x) * torch.cos(
        2 * torch.pi * y) + 4 * mu * torch.pi * torch.pi * torch.sin(2 * torch.pi * x)
          + torch.pi * torch.pi * (3 * mu + lambd) / (mu + lambd) * torch.sin(torch.pi * x) * torch.sin(
                torch.pi * y) - torch.pi * torch.pi * torch.cos(torch.pi * x) * torch.cos(torch.pi * y)
          + alpha * torch.pi * torch.sin(torch.pi * x) * torch.cos(torch.pi * y)) * torch.exp(-t)
    return f2


# boundary condition function
def g(x, y, t, n1, n2):
    g1 = torch.pi * K * torch.cos(torch.pi * x) * torch.sin(torch.pi * y) * torch.exp(-t)
    g2 = torch.pi * K * torch.sin(torch.pi * x) * torch.cos(torch.pi * y) * torch.exp(-t)
    return g1 * n1 + g2 * n2


def h1(x, y, t, n1, n2):
    h11 = (-4 * mu * torch.pi * torch.sin(2 * torch.pi * x) * torch.sin(2 * torch.pi * y)
           + 2 * mu * torch.pi / (mu + lambd) * torch.cos(torch.pi * x) * torch.sin(torch.pi * y)
           - alpha * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
           + torch.pi * lambd / (mu + lambd) * torch.sin(torch.pi * (x + y))) * torch.exp(-t)
    h12 = (2 * mu * torch.pi * (torch.cos(2 * torch.pi * x) - torch.cos(2 * torch.pi * y)) + mu * torch.pi / (
            mu + lambd) * torch.sin(torch.pi * (x + y))) * torch.exp(-t)
    return h11 * n1 + h12 * n2


def h2(x, y, t, n1, n2):
    h21 = (2 * mu * torch.pi * (torch.cos(2 * torch.pi * x) - torch.cos(2 * torch.pi * y)) + mu * torch.pi / (
            mu + lambd) * torch.sin(torch.pi * (x + y))) * torch.exp(-t)
    h22 = (4 * mu * torch.pi * torch.sin(2 * torch.pi * x) * torch.sin(2 * torch.pi * y)
           + 2 * mu * torch.pi / (mu + lambd) * torch.cos(torch.pi * y) * torch.sin(torch.pi * x)
           - alpha * torch.sin(torch.pi * y) * torch.sin(torch.pi * x)
           + torch.pi * lambd / (mu + lambd) * torch.sin(torch.pi * (x + y))) * torch.exp(-t)
    return h21 * n1 + h22 * n2


# PDE loss function for p
def loss_Qs(netp, netp_old, netu_old, x, y, t):
    # the source or sink term Qs is defined here
    # step1
    p = netp(torch.hstack((x, y, t)))  # new p

    p_old = netp_old(torch.hstack((x, y, t)))  # old  p
    u_old = netu_old(torch.hstack((x, y, t)))  # old  u
    u1_old = u_old[:, 0].reshape(-1, 1)
    u2_old = u_old[:, 1].reshape(-1, 1)

    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), retain_graph=True, create_graph=True)[0]
    p_yy = torch.autograd.grad(p_y, y, grad_outputs=torch.ones_like(p_y), retain_graph=True, create_graph=True)[0]

    p_old_t = torch.autograd.grad(p_old, t, grad_outputs=torch.ones_like(p_old), retain_graph=True, create_graph=True)[0]
    u1_old_x = torch.autograd.grad(u1_old, x, grad_outputs=torch.ones_like(u1_old), retain_graph=True, create_graph=True)[0]
    u2_old_y = torch.autograd.grad(u2_old, y, grad_outputs=torch.ones_like(u2_old), retain_graph=True, create_graph=True)[0]
    u1_old_xt = torch.autograd.grad(u1_old_x, t, grad_outputs=torch.ones_like(u1_old_x), retain_graph=True, create_graph=True)[0]
    u2_old_yt = torch.autograd.grad(u2_old_y, t, grad_outputs=torch.ones_like(u2_old_y), retain_graph=True, create_graph=True)[0]

    return (c0 + betaFS) * p_t - K * (p_xx + p_yy) + alpha * (u1_old_xt + u2_old_yt) - betaFS * p_old_t - Qs(x, y, t)


# PDE loss function for u
def loss_f(netu, netp, x, y, t):
    # step 2
    u = netu(torch.hstack((x, y, t)))
    u1 = u[:, 0].reshape(-1, 1)
    u2 = u[:, 1].reshape(-1, 1)

    p = netp(torch.hstack((x, y, t)))

    u1_x = torch.autograd.grad(u1, x, grad_outputs=torch.ones_like(u1), retain_graph=True, create_graph=True)[0]
    u1_y = torch.autograd.grad(u1, y, grad_outputs=torch.ones_like(u1), retain_graph=True, create_graph=True)[0]
    u1_xx = torch.autograd.grad(u1_x, x, grad_outputs=torch.ones_like(u1_x), retain_graph=True, create_graph=True)[0]
    u1_xy = torch.autograd.grad(u1_x, y, grad_outputs=torch.ones_like(u1_x), retain_graph=True, create_graph=True)[0]
    u1_yx = torch.autograd.grad(u1_y, x, grad_outputs=torch.ones_like(u1_y), retain_graph=True, create_graph=True)[0]
    u1_yy = torch.autograd.grad(u1_y, y, grad_outputs=torch.ones_like(u1_y), retain_graph=True, create_graph=True)[0]

    u2_x = torch.autograd.grad(u2, x, grad_outputs=torch.ones_like(u2), retain_graph=True, create_graph=True)[0]
    u2_y = torch.autograd.grad(u2, y, grad_outputs=torch.ones_like(u2), retain_graph=True, create_graph=True)[0]
    u2_xx = torch.autograd.grad(u2_x, x, grad_outputs=torch.ones_like(u2_x), retain_graph=True, create_graph=True)[0]
    u2_xy = torch.autograd.grad(u2_x, y, grad_outputs=torch.ones_like(u2_x), retain_graph=True, create_graph=True)[0]
    u2_yx = torch.autograd.grad(u2_y, x, grad_outputs=torch.ones_like(u2_y), retain_graph=True, create_graph=True)[0]
    u2_yy = torch.autograd.grad(u2_y, y, grad_outputs=torch.ones_like(u2_y), retain_graph=True, create_graph=True)[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

    Lossf1 = - mu * (2 * u1_xx + u1_yy + u2_xy) - lambd * (u1_xx + u2_yx) + alpha * p_x - f1(x, y, t)
    Lossf2 = - mu * (2 * u2_yy + u2_xx + u1_yx) - lambd * (u1_xy + u2_yy) + alpha * p_y - f2(x, y, t)
    return torch.hstack((Lossf1, Lossf2))


# boundary condition function
# for p
def BC_Right_p(netp, x, y, t):
    n1 = 1
    n2 = 0
    p = netp(torch.hstack((x, y, t)))
    Loss_p = p - sol_p(x, y, t)

    return Loss_p


def BC_Left_p(netp, x, y, t):
    n1 = -1
    n2 = 0
    p = netp(torch.hstack((x, y, t)))
    Loss_p = p - sol_p(x, y, t)

    return Loss_p


def BC_Upper_p(netp, x, y, t):
    n1 = 0
    n2 = 1
    p = netp(torch.hstack((x, y, t)))
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    # Loss_p boundary condition residual when move all items to the left, default equal to 0
    Loss_p = p_x * n1 + p_y * n2 - g(x, y, t, n1, n2)

    return Loss_p


def BC_Lower_p(netp, x, y, t):
    n1 = 0
    n2 = -1
    p = netp(torch.hstack((x, y, t)))
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    # Loss_p boundary condition residual when move all items to the left, default equal to 0
    Loss_p = p_x * n1 + p_y * n2 - g(x, y, t, n1, n2)

    return Loss_p


def BC_Bottom_p(netp, x, y, t):
    # initial condition
    p = netp(torch.hstack((x, y, t)))
    Loss_p = p - sol_p(x, y, t)

    return Loss_p


# for u
def BC_Right_u(netp, netu, x, y, t):
    n1 = 1
    n2 = 0
    u = netu(torch.hstack((x, y, t)))
    u1 = u[:, 0].reshape(-1, 1)
    u2 = u[:, 1].reshape(-1, 1)

    # Loss_u1 and Loss_u2 denote the boundary condition when move all items to left, default equal to 0
    Loss_u1 = u1 - sol_u1(x, y, t)
    Loss_u2 = u2 - sol_u2(x, y, t)
    return torch.hstack((Loss_u1, Loss_u2))


def BC_Left_u(netp, netu, x, y, t):
    n1 = -1
    n2 = 0
    u = netu(torch.hstack((x, y, t)))
    u1 = u[:, 0].reshape(-1, 1)
    u2 = u[:, 1].reshape(-1, 1)

    # Loss_u1 and Loss_u2 denote the boundary condition when move all items to left, default equal to 0
    Loss_u1 = u1 - sol_u1(x, y, t)
    Loss_u2 = u2 - sol_u2(x, y, t)
    return torch.hstack((Loss_u1, Loss_u2))


def BC_Upper_u(netp, netu, x, y, t):
    n1 = 0
    n2 = 1
    u = netu(torch.hstack((x, y, t)))
    u1 = u[:, 0].reshape(-1, 1)
    u2 = u[:, 1].reshape(-1, 1)
    p = netp(torch.hstack((x, y, t)))

    u1_x = torch.autograd.grad(u1, x, grad_outputs=torch.ones_like(u1), retain_graph=True, create_graph=True)[0]
    u1_y = torch.autograd.grad(u1, y, grad_outputs=torch.ones_like(u1), retain_graph=True, create_graph=True)[0]
    u2_x = torch.autograd.grad(u2, x, grad_outputs=torch.ones_like(u2), retain_graph=True, create_graph=True)[0]
    u2_y = torch.autograd.grad(u2, y, grad_outputs=torch.ones_like(u2), retain_graph=True, create_graph=True)[0]

    # Loss_u1 and Loss_u2 denote the boundary condition when move all items to left, default equal to 0
    Loss_u1 = n1 * (lambd * (u1_x + u2_y) + 2 * mu * u1_x) + n2 * mu * (u2_x + u1_y) - n1 * alpha * p - h1(x, y, t, n1,
                                                                                                           n2)
    Loss_u2 = n1 * mu * (u2_x + u1_y) + n2 * (lambd * (u2_y + u1_x) + 2 * mu * u2_y) - n2 * alpha * p - h2(x, y, t, n1,
                                                                                                           n2)

    return torch.hstack((Loss_u1, Loss_u2))


def BC_Lower_u(netp, netu, x, y, t):
    n1 = 0
    n2 = -1
    u = netu(torch.hstack((x, y, t)))
    u1 = u[:, 0].reshape(-1, 1)
    u2 = u[:, 1].reshape(-1, 1)
    p = netp(torch.hstack((x, y, t)))

    u1_x = torch.autograd.grad(u1, x, grad_outputs=torch.ones_like(u1), retain_graph=True, create_graph=True)[0]
    u1_y = torch.autograd.grad(u1, y, grad_outputs=torch.ones_like(u1), retain_graph=True, create_graph=True)[0]
    u2_x = torch.autograd.grad(u2, x, grad_outputs=torch.ones_like(u2), retain_graph=True, create_graph=True)[0]
    u2_y = torch.autograd.grad(u2, y, grad_outputs=torch.ones_like(u2), retain_graph=True, create_graph=True)[0]

    # Loss_u1 and Loss_u2 denote the boundary condition when move all items to left, default equal to 0
    Loss_u1 = n1 * (lambd * (u1_x + u2_y) + 2 * mu * u1_x) + n2 * mu * (u2_x + u1_y) - n1 * alpha * p - h1(x, y, t, n1,
                                                                                                           n2)
    Loss_u2 = n1 * mu * (u2_x + u1_y) + n2 * (lambd * (u2_y + u1_x) + 2 * mu * u2_y) - n2 * alpha * p - h2(x, y, t, n1,
                                                                                                           n2)

    return torch.hstack((Loss_u1, Loss_u2))


def BC_Bottom_u(netp, netu, x, y, t):
    # initial condition
    u = netu(torch.hstack((x, y, t)))
    u1 = u[:, 0].reshape(-1, 1)
    u2 = u[:, 1].reshape(-1, 1)
    u10 = sol_u1(x, y, t)
    u20 = sol_u2(x, y, t)
    Loss_u1 = u1 - u10
    Loss_u2 = u2 - u20
    return torch.hstack((Loss_u1, Loss_u2))


"""
4. Training model 
"""

# Data points
N_point = 10000
N_right = 200
N_left = 200
N_upper = 200
N_lower = 200
N_bottom = 500

# iteration setting
out_iter = 50
in_iter = 400

# learning rate setting
decreasing_FS = 1
LR_FS = 0.0001

# loss function
Loss = torch.nn.MSELoss()

# generating network and netu/netp initial and netu_old/netp_old zero initial
netu = Netu()
netu_old = Netu()
netp = Netp()
netp_old = Netp()
netu.to(device)
netp.to(device)
netu_old.to(device)
netp_old.to(device)

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

# Data points used to calculate the L2-error#
x = torch.linspace(0, 1, 500)
y = torch.linspace(0, 1, 500)
m, n = x.shape[0], y.shape[0]
X, Y = torch.meshgrid(x, y)
point = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
t = torch.ones((m * n, 1)) * T_final
inputs = torch.hstack((point, t)).to(device)
Pex = sol_p(inputs[:, 0], inputs[:, 1], inputs[:, 2]).reshape(m, n).cpu().detach().numpy()
u1_ex = sol_u1(inputs[:, 0], inputs[:, 1], inputs[:, 2]).reshape(m, n).cpu().detach().numpy()
u2_ex = sol_u2(inputs[:, 0], inputs[:, 1], inputs[:, 2]).reshape(m, n).cpu().detach().numpy()

# recording list
L2_list = []

# weight option
weigh1 = [1, 1]
weigh2 = [1, 1]

# training
for epoch in range(1, out_iter + 1):

    # Data
    # Boundary condition
    x_Right = np.ones((N_right, 1))  # 1
    y_Right = np.random.rand(N_right, 1)  # y
    t_Right = T_final * np.random.rand(N_right, 1)

    x_Left = np.zeros((N_left, 1))  # 0
    y_Left = np.random.rand(N_right, 1)  # y
    t_Left = T_final * np.random.rand(N_left, 1)

    x_Upper = np.random.rand(N_upper, 1)  # x
    y_Upper = np.ones((N_upper, 1))  # 1
    t_Upper = T_final * np.random.rand(N_upper, 1)

    x_Lower = np.random.rand(N_lower, 1)  # x
    y_Lower = np.zeros((N_lower, 1))  # 0
    t_Lower = T_final * np.random.rand(N_lower, 1)

    x_Bottom = np.random.rand(N_bottom, 1)  # x
    y_Bottom = np.random.rand(N_bottom, 1)  # y
    t_Bottom = np.zeros((N_bottom, 1))

    Right = np.hstack((x_Right, y_Right, t_Right))
    Left = np.hstack((x_Left, y_Left, t_Left))
    Upper = np.hstack((x_Upper, y_Upper, t_Upper))
    Lower = np.hstack((x_Lower, y_Lower, t_Lower))
    Bottom = np.hstack((x_Bottom, y_Bottom, t_Bottom))

    # torch  variable
    torch_x_Right = Variable(torch.from_numpy(x_Right).float(), requires_grad=True).to(device)
    torch_y_Right = Variable(torch.from_numpy(y_Right).float(), requires_grad=True).to(device)
    torch_t_Right = Variable(torch.from_numpy(t_Right).float(), requires_grad=True).to(device)

    torch_x_Left = Variable(torch.from_numpy(x_Left).float(), requires_grad=True).to(device)
    torch_y_Left = Variable(torch.from_numpy(y_Left).float(), requires_grad=True).to(device)
    torch_t_Left = Variable(torch.from_numpy(t_Left).float(), requires_grad=True).to(device)

    torch_x_Upper = Variable(torch.from_numpy(x_Upper).float(), requires_grad=True).to(device)
    torch_y_Upper = Variable(torch.from_numpy(y_Upper).float(), requires_grad=True).to(device)
    torch_t_Upper = Variable(torch.from_numpy(t_Upper).float(), requires_grad=True).to(device)

    torch_x_Lower = Variable(torch.from_numpy(x_Lower).float(), requires_grad=True).to(device)
    torch_y_Lower = Variable(torch.from_numpy(y_Lower).float(), requires_grad=True).to(device)
    torch_t_Lower = Variable(torch.from_numpy(t_Lower).float(), requires_grad=True).to(device)

    torch_x_Bottom = Variable(torch.from_numpy(x_Bottom).float(), requires_grad=True).to(device)
    torch_y_Bottom = Variable(torch.from_numpy(y_Bottom).float(), requires_grad=True).to(device)
    torch_t_Bottom = Variable(torch.from_numpy(t_Bottom).float(), requires_grad=True).to(device)

    # PDE  collocation point
    x_coll = np.random.rand(N_point, 1)
    y_coll = np.random.rand(N_point, 1)
    t_coll = T_final * np.random.rand(N_point, 1)

    torch_x_coll = Variable(torch.from_numpy(x_coll).float(), requires_grad=True).to(device)
    torch_y_coll = Variable(torch.from_numpy(x_coll).float(), requires_grad=True).to(device)
    torch_t_coll = Variable(torch.from_numpy(t_coll).float(), requires_grad=True).to(device)

    # step1
    optimizer1 = optim.Adam(netp.parameters(), lr=LR_FS)  # for netp
    scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer1, decreasing_FS)
    start1 = time.perf_counter()
    for i in range(1, in_iter + 1):
        optimizer1.zero_grad()
        # BC Loss, Res_Right_p residual of boundary condition on Right of p 
        Res_Right_p = BC_Right_p(netp, torch_x_Right, torch_y_Right, torch_t_Right)
        Res_Left_p = BC_Left_p(netp, torch_x_Left, torch_y_Left, torch_t_Left)
        Res_Upper_p = BC_Upper_p(netp, torch_x_Upper, torch_y_Upper, torch_t_Upper)
        Res_Lower_p = BC_Lower_p(netp, torch_x_Lower, torch_y_Lower, torch_t_Lower)
        Res_Bottom_p = BC_Bottom_p(netp, torch_x_Bottom, torch_y_Bottom, torch_t_Bottom)

        mse_Right_p = Loss(Res_Right_p, torch.zeros_like(Res_Right_p))
        mse_Left_p = Loss(Res_Left_p, torch.zeros_like(Res_Left_p))
        mse_Upper_p = Loss(Res_Upper_p, torch.zeros_like(Res_Upper_p))
        mse_Lower_p = Loss(Res_Lower_p, torch.zeros_like(Res_Lower_p))
        mse_Bottom_p = Loss(Res_Bottom_p, torch.zeros_like(Res_Bottom_p))

        mse_p = mse_Right_p + mse_Left_p + mse_Upper_p + mse_Lower_p + mse_Bottom_p

        # PDE Loss
        LHS1_coll = loss_Qs(netp, netp_old, netu_old, torch_x_coll, torch_y_coll,
                            torch_t_coll)  # step1  right hand side
        RHS1_coll = torch.zeros_like(LHS1_coll).to(device)
        mse_fp = Loss(LHS1_coll, RHS1_coll)

        # loss = mse_u + mse_f
        loss_step1 = weigh1[0] * mse_p + weigh1[1] * mse_fp

        # backward
        loss_step1.backward()
        optimizer1.step()
        scheduler1.step()

    end1 = time.perf_counter()

    # step2
    optimizer2 = optim.Adam(netu.parameters(), lr=LR_FS)  # for netu
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer2, decreasing_FS)
    start2 = time.perf_counter()
    for i in range(1, in_iter + 1):
        optimizer2.zero_grad()
        # BC Loss, Res_Right_u residual of boundary condition on Right of u 
        Res_Right_u = BC_Right_u(netp, netu, torch_x_Right, torch_y_Right, torch_t_Right)
        Res_Left_u = BC_Left_u(netp, netu, torch_x_Left, torch_y_Left, torch_t_Left)
        Res_Upper_u = BC_Upper_u(netp, netu, torch_x_Upper, torch_y_Upper, torch_t_Upper)
        Res_Lower_u = BC_Lower_u(netp, netu, torch_x_Lower, torch_y_Lower, torch_t_Lower)
        Res_Bottom_u = BC_Bottom_u(netp, netu, torch_x_Bottom, torch_y_Bottom, torch_t_Bottom)

        mse_Right_u = Loss(Res_Right_u, torch.zeros_like(Res_Right_u))
        mse_Left_u = Loss(Res_Left_u, torch.zeros_like(Res_Left_u))
        mse_Upper_u = Loss(Res_Upper_u, torch.zeros_like(Res_Upper_u))
        mse_Lower_u = Loss(Res_Lower_u, torch.zeros_like(Res_Lower_u))
        mse_Bottom_u = Loss(Res_Bottom_u, torch.zeros_like(Res_Bottom_u))

        mse_u = mse_Right_u + mse_Left_u + mse_Upper_u + mse_Lower_u + mse_Bottom_u

        #  PDE collection point num x 3(dim) -> num x 2(dim) 
        LHS2_coll = loss_f(netu, netp, torch_x_coll, torch_y_coll, torch_t_coll)
        RHS2_coll = torch.zeros_like(LHS2_coll).to(device)
        mse_fu = Loss(LHS2_coll, RHS2_coll)

        # loss = mse_u + mse_f
        loss_step2 = weigh2[0] * mse_u + weigh2[1] * mse_fu

        # backward    
        loss_step2.backward()
        optimizer2.step()
        scheduler2.step()

    end2 = time.perf_counter()

    # updating netp_old and  netu_old
    netp_old = deepcopy(netp)
    netu_old = deepcopy(netu)

    # recording
    Time1 = end1 - start1 + end2 - start2
    print('iteration:{0:5d}, loss1: {1:14.12f} loss2: {2:14.12f} Time:{3:14.12f} '
          .format(epoch, mse_p + mse_fp, mse_u + mse_fu, Time1))

    # L2-error of p and u  at every iteration
    outp = netp(inputs).reshape(m, n).cpu().detach().numpy()
    outu = netu(inputs).cpu().detach().numpy()
    outu1 = outu[:, 0].reshape(m, n)
    outu2 = outu[:, 1].reshape(m, n)

    P_absolute_error = abs(Pex - outp).reshape(-1, 1)
    u1_absolute_error = abs(u1_ex - outu1).reshape(-1, 1)
    u2_absolute_error = abs(u2_ex - outu2).reshape(-1, 1)

    l2_errorp = (sum(P_absolute_error ** 2 / (m * n)))[0]
    l2_erroru1 = (sum(u1_absolute_error ** 2 / (m * n)))[0]
    l2_erroru2 = (sum(u2_absolute_error ** 2 / (m * n)))[0]
    L2_list.append([epoch, l2_errorp, l2_erroru1, l2_erroru2])

# MSE in every iteration
print('MSE for p,u1 and u2.')
for i in L2_list:
    print('iteration:{0:3d}, p:{1:8f}, u1:{2:8f}, u2:{3:8f}, Total:{4:8f}'.format(i[0], i[1], i[2], i[3],
                                                                                  i[1] + i[2] + i[3]))

"""
5. Visualization
"""
#  picture points
outp = netp(inputs).reshape(m, n).cpu().detach().numpy()
outu = netu(inputs).cpu().detach().numpy()
outu1 = outu[:, 0].reshape(m, n)
outu2 = outu[:, 1].reshape(m, n)
Xnp = X.detach().numpy()
Ynp = Y.detach().numpy()

Pex = sol_p(inputs[:, 0], inputs[:, 1], inputs[:, 2]).reshape(m, n).cpu().detach().numpy()
u1_ex = sol_u1(inputs[:, 0], inputs[:, 1], inputs[:, 2]).reshape(m, n).cpu().detach().numpy()
u2_ex = sol_u2(inputs[:, 0], inputs[:, 1], inputs[:, 2]).reshape(m, n).cpu().detach().numpy()

# vector picture points
N = 15
x = torch.linspace(0, 1, N)
y = torch.linspace(0, 1, N)
X2, Y2 = torch.meshgrid(x, y)
point = torch.hstack((X2.reshape(-1, 1), Y2.reshape(-1, 1)))
t = torch.ones((N * N, 1)) * T_final
inputs = torch.hstack((point, t)).to(device)
XX2 = X2.detach().numpy()
YY2 = Y2.detach().numpy()
pt_x_vector = Variable(torch.from_numpy(XX2.reshape(-1, 1)).float(), requires_grad=True).to(device)
pt_y_vector = Variable(torch.from_numpy(YY2.reshape(-1, 1)).float(), requires_grad=True).to(device)
pt_t_vector = Variable(torch.from_numpy(T_final * np.ones((N * N, 1))).float(), requires_grad=True).to(device)

output = netp(torch.hstack((pt_x_vector, pt_y_vector, pt_t_vector)))
zz = output.cpu()
output = netu(torch.hstack((pt_x_vector, pt_y_vector, pt_t_vector)))
zz = output.cpu()
p_fs_vector = zz[:, 0].detach().numpy().reshape(N, N)
u1_fs_vector = zz[:, 0].detach().numpy().reshape(N, N)
u2_fs_vector = zz[:, 1].detach().numpy().reshape(N, N)

u1_ex_vector = sol_u1(pt_x_vector, pt_y_vector, pt_t_vector).reshape(N, N).cpu().detach().numpy()
u2_ex_vector = sol_u2(pt_x_vector, pt_y_vector, pt_t_vector).reshape(N, N).cpu().detach().numpy()

# figure
fig, ax = plt.subplots()
cs = ax.contourf(Xnp, Ynp, outp)
plt.quiver(XX2, YY2, u1_fs_vector, u2_fs_vector, headwidth=3, alpha=1)
cs.cmap.set_under('blue')
cs.changed()
cbar = fig.colorbar(cs)
plt.title('FS-PINN solution')

fig, ax = plt.subplots()
cs = ax.contourf(Xnp, Ynp, Pex)
plt.quiver(XX2, YY2, u1_ex_vector, u2_ex_vector, headwidth=3, alpha=1)
cbar = fig.colorbar(cs)
plt.title('Analytical solution')

plt.show()
