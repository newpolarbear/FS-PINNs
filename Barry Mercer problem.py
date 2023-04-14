# -*- coding: utf-8 -*-
"""
@author: Huipeng Gu, Pengxiang Hong.
The benchmark test is Barry Mercer problem.
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
Lx = 1
Ly = 1
x0 = 0.25
y0 = 0.25
Guass_alpha = 0.04

beta = 2.0
alpha = 1.0
K = 1.0
c0 = 0.0
lambd = 0.1
mu = 0.2
eta = 1.5
betaFS = alpha ** 2 / (lambd + mu)
T_final = 0.5 * np.pi / beta

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


# model function
def Qs(x, y, t):
    return beta * torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (Guass_alpha ** 2)) * torch.sin(beta * t) / (
            torch.pi * Guass_alpha ** 2)


def EQforP(netp, x, y, t):
    # step1
    p = netp(torch.hstack((x, y, t)))

    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), retain_graph=True, create_graph=True)[0]
    p_yy = torch.autograd.grad(p_y, y, grad_outputs=torch.ones_like(p_y), retain_graph=True, create_graph=True)[0]

    Loss = (c0 + betaFS) * (lambd + 2 * mu) * p_t - (p_xx + p_yy)
    return Loss


# right hand side equation for p at step1
def step1old(netp_old, netu_old, x, y, t):
    # return step1's  RHS
    # step1
    p_old = netp_old(torch.hstack((x, y, t)))  # old  p
    u_old = netu_old(torch.hstack((x, y, t)))  # old  u
    u1_old = u_old[:, 0].reshape(-1, 1)
    u2_old = u_old[:, 1].reshape(-1, 1)

    p_old_t = torch.autograd.grad(p_old, t, grad_outputs=torch.ones_like(p_old), retain_graph=True, create_graph=True)[0]
    u1_old_x = torch.autograd.grad(u1_old, x, grad_outputs=torch.ones_like(u1_old), retain_graph=True, create_graph=True)[0]
    u2_old_y = torch.autograd.grad(u2_old, y, grad_outputs=torch.ones_like(u2_old), retain_graph=True, create_graph=True)[0]
    u1_old_xt = torch.autograd.grad(u1_old_x, t, grad_outputs=torch.ones_like(u1_old_x), retain_graph=True, create_graph=True)[0]
    u2_old_yt = torch.autograd.grad(u2_old_y, t, grad_outputs=torch.ones_like(u2_old_y), retain_graph=True, create_graph=True)[0]

    return Qs(x, y, t) - alpha * (u1_old_xt + u2_old_yt) + betaFS * (lambd + 2 * mu) * p_old_t


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

    return Qs(x, y, t) - alpha * (u1_old_xt + u2_old_yt) + betaFS * (lambd + 2 * mu) * p_old_t - (c0 + betaFS) * (lambd + 2 * mu) * p_t + (p_xx + p_yy)


# step 2 equation for u
def loss_f(netu, netp, x, y, t):
    # step 2
    u = netu(torch.hstack((x, y, t)))  # new u
    u1 = u[:, 0].reshape(-1, 1)
    u2 = u[:, 1].reshape(-1, 1)

    p = netp(torch.hstack((x, y, t)))  # new p

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

    Lossf1 = (eta + 1) * u1_xx + u1_yy + eta / 2 * (u2_xy + u2_yx) - (eta + 1) * p_x
    Lossf2 = u2_xx + (eta + 1) * u2_yy + eta / 2 * (u1_yx + u1_xy) - (eta + 1) * p_y
    return torch.hstack((Lossf1, Lossf2))


# boundary condition function
# for p
def BC_Right_p(netp, x, y, t):
    p = netp(torch.hstack((x, y, t)))
    Loss_p = p

    return Loss_p


def BC_Left_p(netp, x, y, t):
    p = netp(torch.hstack((x, y, t)))

    Loss_p = p
    return Loss_p


def BC_Upper_p(netp, x, y, t):
    p = netp(torch.hstack((x, y, t)))
    Loss_p = p

    return Loss_p


def BC_Lower_p(netp, x, y, t):
    p = netp(torch.hstack((x, y, t)))
    Loss_p = p

    return Loss_p


def BC_Bottom_p(netp, x, y, t):
    # initial condition
    p = netp(torch.hstack((x, y, t)))
    Loss_p = p

    return Loss_p


# for u
def BC_Right_u(netp, netu, x, y, t):
    u = netu(torch.hstack((x, y, t)))
    u1 = u[:, 0].reshape(-1, 1)
    u2 = u[:, 1].reshape(-1, 1)
    u1_x = torch.autograd.grad(u1, x, grad_outputs=torch.ones_like(u1), retain_graph=True, create_graph=True)[0]
    Loss_u1 = u1_x
    Loss_u2 = u2

    return torch.hstack((Loss_u1, Loss_u2))


def BC_Left_u(netp, netu, x, y, t):
    u = netu(torch.hstack((x, y, t)))
    u1 = u[:, 0].reshape(-1, 1)
    u2 = u[:, 1].reshape(-1, 1)
    u1_x = torch.autograd.grad(u1, x, grad_outputs=torch.ones_like(u1), retain_graph=True, create_graph=True)[0]
    Loss_u1 = u1_x
    Loss_u2 = u2

    return torch.hstack((Loss_u1, Loss_u2))


def BC_Upper_u(netp, netu, x, y, t):
    u = netu(torch.hstack((x, y, t)))
    u1 = u[:, 0].reshape(-1, 1)
    u2 = u[:, 1].reshape(-1, 1)
    u2_y = torch.autograd.grad(u2, y, grad_outputs=torch.ones_like(u2), retain_graph=True, create_graph=True)[0]
    Loss_u1 = u1
    Loss_u2 = u2_y

    return torch.hstack((Loss_u1, Loss_u2))


def BC_Lower_u(netp, netu, x, y, t):
    u = netu(torch.hstack((x, y, t)))
    u1 = u[:, 0].reshape(-1, 1)
    u2 = u[:, 1].reshape(-1, 1)
    u2_y = torch.autograd.grad(u2, y, grad_outputs=torch.ones_like(u2), retain_graph=True, create_graph=True)[0]
    Loss_u1 = u1
    Loss_u2 = u2_y

    return torch.hstack((Loss_u1, Loss_u2))


def BC_Bottom_u(netp, netu, x, y, t):
    # initial condition
    u = netu(torch.hstack((x, y, t)))
    u1 = u[:, 0].reshape(-1, 1)
    u2 = u[:, 1].reshape(-1, 1)
    Loss_u1 = u1
    Loss_u2 = u2

    return torch.hstack((Loss_u1, Loss_u2))


# Analytical solution
def lamb(n):
    return np.pi * n


def sol(x, y, t):
    num = 30
    p = 0
    u1 = 0
    u2 = 0
    p_hat = np.zeros([num, num])
    u1_hat = np.zeros([num, num])
    u2_hat = np.zeros([num, num])
    for n in range(num):
        for q in range(num):
            lamb_nq = lamb(n + 1) ** 2 + lamb(q + 1) ** 2
            p_hat[n, q] = -2.0 * np.sin(lamb(n + 1) * x0) * np.sin(lamb(q + 1) * y0) / (lamb_nq ** 2 + 1.0) * (
                        lamb_nq * np.sin(beta * t) - np.cos(beta * t) + np.exp(-lamb_nq * beta * t))
            u1_hat[n, q] = lamb(n + 1) / lamb_nq * p_hat[n, q]
            u2_hat[n, q] = lamb(q + 1) / lamb_nq * p_hat[n, q]
    for n in range(num):
        for q in range(num):
            p = p + p_hat[n, q] * np.sin(lamb(n + 1) * x) * np.sin(lamb(q + 1) * y)
            u1 = u1 + u1_hat[n, q] * np.cos(lamb(n + 1) * x) * np.sin(lamb(q + 1) * y)
            u2 = u2 + u2_hat[n, q] * np.sin(lamb(n + 1) * x) * np.cos(lamb(q + 1) * y)
    return -4 * p, 4 * u1, 4 * u2


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

# weight option
weight1 = [1000, 1]
weight2 = [100, 1]

# recording list
Time_Sum_FS = 0

# training
for epoch in range(1, out_iter + 1):

    # Data
    # Boundary condition
    Right = np.hstack((np.ones((N_right, 1), dtype=float),
                       np.random.rand(N_right, 1),
                       T_final * np.random.rand(N_right, 1)))
    Left = np.hstack((np.zeros((N_left, 1), dtype=float),
                      np.random.rand(N_left, 1),
                      T_final * np.random.rand(N_left, 1)))
    Upper = np.hstack((np.random.rand(N_upper, 1),
                       np.ones((N_upper, 1), dtype=float),
                       T_final * np.random.rand(N_upper, 1)))
    Lower = np.hstack((np.random.rand(N_lower, 1),
                       np.zeros((N_lower, 1), dtype=float),
                       T_final * np.random.rand(N_lower, 1)))
    Bottom = np.hstack((np.random.rand(N_bottom, 1),
                        np.random.rand(N_bottom, 1),
                        np.zeros((N_bottom, 1), dtype=float)))

    # torch variable
    torch_x_Left = Variable(torch.from_numpy(Left[:, 0].reshape(-1, 1)).float(), requires_grad=True).to(device)
    torch_y_Left = Variable(torch.from_numpy(Left[:, 1].reshape(-1, 1)).float(), requires_grad=True).to(device)
    torch_t_Left = Variable(torch.from_numpy(Left[:, 2].reshape(-1, 1)).float(), requires_grad=True).to(device)

    torch_x_Right = Variable(torch.from_numpy(Right[:, 0].reshape(-1, 1)).float(), requires_grad=True).to(device)
    torch_y_Right = Variable(torch.from_numpy(Right[:, 1].reshape(-1, 1)).float(), requires_grad=True).to(device)
    torch_t_Right = Variable(torch.from_numpy(Right[:, 2].reshape(-1, 1)).float(), requires_grad=True).to(device)

    torch_x_Lower = Variable(torch.from_numpy(Lower[:, 0].reshape(-1, 1)).float(), requires_grad=True).to(device)
    torch_y_Lower = Variable(torch.from_numpy(Lower[:, 1].reshape(-1, 1)).float(), requires_grad=True).to(device)
    torch_t_Lower = Variable(torch.from_numpy(Lower[:, 2].reshape(-1, 1)).float(), requires_grad=True).to(device)

    torch_x_Upper = Variable(torch.from_numpy(Upper[:, 0].reshape(-1, 1)).float(), requires_grad=True).to(device)
    torch_y_Upper = Variable(torch.from_numpy(Upper[:, 1].reshape(-1, 1)).float(), requires_grad=True).to(device)
    torch_t_Upper = Variable(torch.from_numpy(Upper[:, 2].reshape(-1, 1)).float(), requires_grad=True).to(device)

    torch_x_Bottom = Variable(torch.from_numpy(Bottom[:, 0].reshape(-1, 1)).float(), requires_grad=True).to(device)
    torch_y_Bottom = Variable(torch.from_numpy(Bottom[:, 1].reshape(-1, 1)).float(), requires_grad=True).to(device)
    torch_t_Bottom = Variable(torch.from_numpy(Bottom[:, 2].reshape(-1, 1)).float(), requires_grad=True).to(device)

    # PDE  collocation points
    x_coll = np.random.uniform(low=0.0, high=1.0, size=(N_point, 1))
    y_coll = np.random.uniform(low=0.0, high=1.0, size=(N_point, 1))
    t_coll = T_final * np.random.uniform(low=0.0, high=1.0, size=(N_point, 1))

    torch_x_coll = Variable(torch.from_numpy(x_coll).float(), requires_grad=True).to(device)
    torch_y_coll = Variable(torch.from_numpy(y_coll).float(), requires_grad=True).to(device)
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
        LHS1_coll = loss_Qs(netp, netp_old, netu_old, torch_x_coll, torch_y_coll,torch_t_coll)
        RHS1_coll = torch.zeros_like(LHS1_coll).to(device)
        mse_fp = Loss(LHS1_coll, RHS1_coll)

        # loss = mse_u + mse_f
        loss_step1 = weight1[0] * (mse_Right_p + mse_Left_p + mse_Upper_p + mse_Lower_p + mse_Bottom_p)+ weight1[1] * mse_fp

        # backward
        loss_step1.backward()
        optimizer1.step()
        scheduler1.step()
    end1 = time.perf_counter()

    # step2
    optimizer2 = optim.Adam(netu.parameters(), lr=LR_FS)
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

        #  PDE Loss
        LHS2_coll = loss_f(netu, netp, torch_x_coll, torch_y_coll, torch_t_coll)
        RHS2_coll = torch.zeros_like(LHS2_coll).to(device)
        mse_fu = Loss(LHS2_coll, RHS2_coll)

        # loss = mse_u + mse_f
        loss_step2 = weight2[0] * (mse_Right_u + mse_Left_u + mse_Upper_u + mse_Lower_u + mse_Bottom_u) + weight2[1] * mse_fu

        # backward
        loss_step2.backward()
        optimizer2.step()
        scheduler2.step()

    end2 = time.perf_counter()

    # updating netp_old and  netu_old
    netp_old = deepcopy(netp)
    netu_old = deepcopy(netu)

    Time1 = end1 - start1 + end2 - start2
    Time_Sum_FS = Time_Sum_FS + Time1
    print('iteration:{0:5d} loss1: {1:14.12f} loss2: {2:14.12f} Time:{3:14.12f} '
          .format(epoch, mse_p + mse_fp, mse_u + mse_fu, Time1))


"""
5. Visualization
"""
#  picture points
x = torch.linspace(0, 1, 50)
y = torch.linspace(0, 1, 50)
m, n = x.shape[0], y.shape[0]
X, Y = torch.meshgrid(x, y)
point = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
t = torch.ones((m * n, 1)) * T_final
inputs = torch.hstack((point, t)).to(device)
outp = netp(inputs).reshape(m, n).cpu().detach().numpy()
outu = netu(inputs).cpu().detach().numpy()
outu1 = outu[:, 0].reshape(m, n)
outu2 = outu[:, 1].reshape(m, n)
Xnp = X.detach().numpy()
Ynp = Y.detach().numpy()

# vector picture points
N = 15
x = torch.linspace(0, 1, N)
y = torch.linspace(0, 1, N)
X2, Y2 = torch.meshgrid(x, y)
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

# Analytical solution
Tnp = T_final * np.ones_like(X)
p_sol = np.zeros_like(X)
u1_sol = np.zeros_like(X)
u2_sol = np.zeros_like(X)
for i in range(len(X)):
    for j in range(len(X[0])):
        p_sol[i][j], u1_sol[i][j], u2_sol[i][j] = sol(Xnp[i][j], Ynp[i][j], Tnp[i][j])

# Analytical solution vector
Tnp_vector = T_final * np.ones_like(X2)
p_sol_vector = np.zeros_like(X2)
u1_sol_vector = np.zeros_like(X2)
u2_sol_vector = np.zeros_like(X2)
for i in range(len(X2)):
    for j in range(len(X2[0])):
        p_sol_vector[i][j], u1_sol_vector[i][j], u2_sol_vector[i][j] = sol(XX2[i][j], YY2[i][j], Tnp_vector[i][j])

# figure
# p
fig, ax = plt.subplots()
levels = np.arange(-0.1, 1.2, 0.1)
cs = ax.contourf(Xnp, Ynp, outp, levels)
plt.quiver(XX2, YY2, u1_fs_vector, u2_fs_vector, headwidth=3, alpha=1)
cs.cmap.set_under('blue')
cs.changed()
cbar = fig.colorbar(cs)
plt.title('FS-PINN solution')

fig, ax = plt.subplots()
cs = ax.contourf(Xnp, Ynp, p_sol, levels)
plt.quiver(XX2, YY2, u1_sol_vector, u2_sol_vector, headwidth=3, alpha=1)
cs.cmap.set_under('blue')
cs.changed()
cbar = fig.colorbar(cs)
plt.title('Analytical solution')
plt.show()