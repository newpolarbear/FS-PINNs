# -*- coding: utf-8 -*-
"""
@author: Huipeng Gu, Pengxiang Hong.
The benchmark test is testing time of PINN vs FS-PINN.

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
T_final = 0.5

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


class Netn(nn.Module):
    def __init__(self):
        super(Netn, self).__init__()
        self.hidden_layer1 = nn.Linear(3, neurons_u + neurons_p)
        self.hidden_layer2 = nn.Linear(neurons_u + neurons_p, neurons_u + neurons_p)
        self.hidden_layer3 = nn.Linear(neurons_u + neurons_p, neurons_u + neurons_p)
        self.output_layer = nn.Linear(neurons_u + neurons_p, 3)

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
    Qs = (c0 + 2 * K) * torch.sin(x + y) * torch.exp(t) + alpha * (x + y)

    output_netp = netp(torch.hstack((x, y, t)))
    output_netu_old = netu_old(torch.hstack((x, y, t)))
    output_netp_old = netp_old(torch.hstack((x, y, t)))

    p = output_netp[:, 0].reshape(-1, 1)
    p_old = output_netp_old[:, 0].reshape(-1, 1)
    u1_old = output_netu_old[:, 0].reshape(-1, 1)
    u2_old = output_netu_old[:, 1].reshape(-1, 1)

    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), retain_graph=True, create_graph=True)[0]
    p_yy = torch.autograd.grad(p_y, y, grad_outputs=torch.ones_like(p_y), retain_graph=True, create_graph=True)[0]

    p_old_t = torch.autograd.grad(p_old, t, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

    u1_old_x = torch.autograd.grad(u1_old, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    u2_old_y = torch.autograd.grad(u2_old, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    u1_old_xt = torch.autograd.grad(u1_old_x, t, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    u2_old_yt = torch.autograd.grad(u2_old_y, t, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

    ret0 = (c0 + betaFS) * p_t - K * (p_xx + p_yy) + alpha * (u1_old_xt + u2_old_yt) - betaFS * p_old_t - Qs

    return ret0

# PDE loss function for u
def loss_f(netu, netp, x, y, t):
    # the body force term f is defined here
    f1 = -(lambd + 2 * mu) * t + alpha * torch.cos(x + y) * torch.exp(t)
    f2 = -(lambd + 2 * mu) * t + alpha * torch.cos(x + y) * torch.exp(t)

    output_netp = netp(torch.hstack((x, y, t)))
    output_netu = netu(torch.hstack((x, y, t)))

    p = output_netp[:, 0].reshape(-1, 1)
    u1 = output_netu[:, 0].reshape(-1, 1)
    u2 = output_netu[:, 1].reshape(-1, 1)

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

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

    ret1 = - mu * (2 * u1_xx + u1_yy + u2_xy) - lambd * (u1_xx + u2_yx) + alpha * p_x - f1
    ret2 = - mu * (2 * u2_yy + u2_xx + u1_yx) - lambd * (u1_xy + u2_yy) + alpha * p_y - f2

    return torch.hstack((ret1, ret2))


# PDE loss function for (p,u)
def loss_N(netn, x, y, t):
    # the body force term f, and the source or sink term f are defined here
    f1 = -(lambd + 2 * mu) * t + alpha * torch.cos(x + y) * torch.exp(t)
    f2 = -(lambd + 2 * mu) * t + alpha * torch.cos(x + y) * torch.exp(t)
    Qs = (c0 + 2 * K) * torch.sin(x + y) * torch.exp(t) + alpha * (x + y)

    output_netn = netn(torch.hstack((x, y, t)))

    u1 = output_netn[:, 0].reshape(-1, 1)
    u2 = output_netn[:, 1].reshape(-1, 1)
    p = output_netn[:, 2].reshape(-1, 1)

    u1_x = torch.autograd.grad(u1, x, grad_outputs=torch.ones_like(u1), retain_graph=True, create_graph=True)[0]
    u1_y = torch.autograd.grad(u1, y, grad_outputs=torch.ones_like(u1), retain_graph=True, create_graph=True)[0]
    u1_xx = torch.autograd.grad(u1_x, x, grad_outputs=torch.ones_like(u1_x), retain_graph=True, create_graph=True)[0]
    u1_xy = torch.autograd.grad(u1_x, y, grad_outputs=torch.ones_like(u1_x), retain_graph=True, create_graph=True)[0]
    u1_yx = torch.autograd.grad(u1_y, x, grad_outputs=torch.ones_like(u1_y), retain_graph=True, create_graph=True)[0]
    u1_yy = torch.autograd.grad(u1_y, y, grad_outputs=torch.ones_like(u1_y), retain_graph=True, create_graph=True)[0]
    u1_xt = torch.autograd.grad(u1_x, t, grad_outputs=torch.ones_like(u1_x), retain_graph=True, create_graph=True)[0]

    u2_x = torch.autograd.grad(u2, x, grad_outputs=torch.ones_like(u2), retain_graph=True, create_graph=True)[0]
    u2_y = torch.autograd.grad(u2, y, grad_outputs=torch.ones_like(u2), retain_graph=True, create_graph=True)[0]
    u2_xx = torch.autograd.grad(u2_x, x, grad_outputs=torch.ones_like(u2_x), retain_graph=True, create_graph=True)[0]
    u2_xy = torch.autograd.grad(u2_x, y, grad_outputs=torch.ones_like(u2_x), retain_graph=True, create_graph=True)[0]
    u2_yx = torch.autograd.grad(u2_y, x, grad_outputs=torch.ones_like(u2_y), retain_graph=True, create_graph=True)[0]
    u2_yy = torch.autograd.grad(u2_y, y, grad_outputs=torch.ones_like(u2_y), retain_graph=True, create_graph=True)[0]
    u2_yt = torch.autograd.grad(u2_y, t, grad_outputs=torch.ones_like(u2_y), retain_graph=True, create_graph=True)[0]

    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_yy = torch.autograd.grad(p_y, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

    ret1 = - mu * (2 * u1_xx + u1_yy + u2_xy) - lambd * (u1_xx + u2_yx) + alpha * p_x - f1
    ret2 = - mu * (2 * u2_yy + u2_xx + u1_yx) - lambd * (u1_xy + u2_yy) + alpha * p_y - f2
    ret3 = c0 * p_t - K * (p_xx + p_yy) + alpha * (u1_xt + u2_yt) - Qs

    return torch.hstack((ret1, ret2, ret3))

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

# recording list
L2_Time_FS = []
Train_Time = []
Time_Sum_PINN = 0
Time_PINN_l2_list = []

# iteration setting
out_iter = 0
in_iter = 0
Sum_iter = 20000
# iteration list of in_iter and out_iter
vary_iter_list = [[[25, 800]],
                  [[100, 200]],
                  [[400, 50]],
                  [[2000, 10]],
                  ]

# learning rate setting
decreasing = 1
LR = 0.0001
decreasing_FS = 1
LR_FS = 0.0001

# loss function
Loss = torch.nn.MSELoss()

# different combo of FS-PINN‘s iteration
for iter_list in vary_iter_list:

    netu = Netu()
    netu_old = Netu()
    netp = Netp()
    netp_old = Netp()
    netn = Netn()
    netu = netu.to(device)
    netu_old = netu_old.to(device)
    netp = netp.to(device)
    netp_old = netp_old.to(device)
    netn = netn.to(device)
    print("\nCase:N_p={0} and N_u={1} with iter list={2}".format(neurons_p, neurons_u, *iter_list))

    """
    network netu/netp/netn initial and netu_old/netp_old zero initial
    """
    for m in netu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
    for m in netp.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
    for m in netn.modules():
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

    Time_Sum = 0
    Time_FS_l2_list = []

    optimizer_n = torch.optim.Adam(netn.parameters(), lr=LR)
    scheduler_n = torch.optim.lr_scheduler.ExponentialLR(optimizer_n, decreasing)

    # Data points used to calculate the L2-error#
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    m, n = x.shape[0], y.shape[0]
    X, Y = np.meshgrid(x, y)
    point = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    t = np.ones((m * n, 1)) * T_final
    inputs = np.hstack((point, t))

    u1_ex = 0.5 * inputs[:, 2] * inputs[:, 0] * inputs[:, 0]
    u2_ex = 0.5 * inputs[:, 2] * inputs[:, 1] * inputs[:, 1]
    p_ex = np.sin(inputs[:, 0] + inputs[:, 1]) * np.exp(inputs[:, 2])

    u1_ex = Variable(torch.from_numpy(u1_ex.reshape(-1, 1)).float(), requires_grad=False).to(device)
    u2_ex = Variable(torch.from_numpy(u2_ex.reshape(-1, 1)).float(), requires_grad=False).to(device)
    p_ex = Variable(torch.from_numpy(p_ex.reshape(-1, 1)).float(), requires_grad=False).to(device)
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
    Time_FS_l2_list = [[0, p_l2.item(), u1_l2.item(), u2_l2.item(), l2_FS.item()]]

    # initial L2-error of PINN
    if iter_list == vary_iter_list[0]:
        output = netn(inputs)
        out_u1 = output[:, 0].reshape(-1, 1)
        out_u2 = output[:, 1].reshape(-1, 1)
        out_p = output[:, 2].reshape(-1, 1)
        p_abs_error = torch.abs(p_ex - out_p).reshape(-1, 1)
        u1_abs_error = torch.abs(u1_ex - out_u1).reshape(-1, 1)
        u2_abs_error = torch.abs(u2_ex - out_u2).reshape(-1, 1)
        p_l2 = torch.sum(p_abs_error ** 2 / (m * n))
        u1_l2 = torch.sum(u1_abs_error ** 2 / (m * n))
        u2_l2 = torch.sum(u2_abs_error ** 2 / (m * n))

        l2 = p_l2 + u1_l2 + u2_l2
        Time_PINN_l2_list = [[0, p_l2.item(), u1_l2.item(), u2_l2.item(), l2.item()]]

    # training
    epoch = 0
    for in_iter, out_iter in iter_list:

        # FS-PINN
        for out_epoch in range(out_iter):

            epoch = epoch + 1
            Time_FS1_Sum = 0
            Time_FS2_Sum = 0
            Time_PINN_Sum = 0

            # Data
            # boundary and initial points
            Right = np.hstack(
                (np.ones((N_right, 1), dtype=float), np.random.rand(N_right, 1), np.random.rand(N_right, 1)))
            Left = np.hstack(
                (np.zeros((N_left, 1), dtype=float), np.random.rand(N_right, 1), np.random.rand(N_right, 1)))
            Upper = np.hstack(
                (np.random.rand(N_upper, 1), np.ones((N_upper, 1), dtype=float), np.random.rand(N_upper, 1)))
            Lower = np.hstack(
                (np.random.rand(N_lower, 1), np.zeros((N_lower, 1), dtype=float), np.random.rand(N_lower, 1)))
            Bottom = np.hstack(
                (np.random.rand(N_bottom, 1), np.random.rand(N_bottom, 1), np.zeros((N_bottom, 1), dtype=float)))
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

            optimizer_p = torch.optim.Adam(netp.parameters(), lr=LR_FS)
            scheduler_p = torch.optim.lr_scheduler.ExponentialLR(optimizer_p, decreasing_FS)

            output = netu(inputs)
            out_u1 = output[:, 0].reshape(-1, 1)
            out_u2 = output[:, 1].reshape(-1, 1)
            u1_abs_error = torch.abs(u1_ex - out_u1).reshape(-1, 1)
            u2_abs_error = torch.abs(u2_ex - out_u2).reshape(-1, 1)
            u1_l2 = torch.sum(u1_abs_error ** 2 / (m * n))
            u2_l2 = torch.sum(u2_abs_error ** 2 / (m * n))

            for in_epoch in range(in_iter):
                start1 = time.perf_counter()
                # train netp using netp_old, netu_old
                optimizer_p.zero_grad()  # to make the gradients zero

                # boundary and initial conditions for netp
                torch_X_train = Variable(torch.from_numpy(X_u_train).float(), requires_grad=False).to(device)
                torch_u_train = Variable(torch.from_numpy(p_train.reshape(-1, 1)).float(), requires_grad=False).to(
                    device)

                net_bc_out1 = netp(torch_X_train)
                mse_p = Loss(net_bc_out1, torch_u_train)

                # PDE loss for netp
                all_zeros = np.zeros((N_point, 1))
                pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)

                pde_p = loss_Qs(netp, netu_old, netp_old, pt_x_collocation, pt_y_collocation, pt_t_collocation)
                mse_Qs = Loss(pde_p, pt_all_zeros)

                # total loss for network p
                loss1 = mse_p + mse_Qs
                loss1.backward()
                optimizer_p.step()
                scheduler_p.step()

                end1 = time.perf_counter()
                #  time X L2-error
                out_p = netp(inputs).reshape(-1, 1)
                p_abs_error = torch.abs(p_ex - out_p).reshape(-1, 1)
                p_l2 = torch.sum(p_abs_error ** 2 / (m * n))
                l2_FS = l2_FS = p_l2 + u1_l2 + u2_l2
                Time1 = end1 - start1
                Time_FS1_Sum = Time_FS1_Sum + Time1
                Time_Sum = Time_Sum + Time1
                Time_FS_l2_list.append([Time_Sum, p_l2.item(), u1_l2.item(), u2_l2.item(), l2_FS.item()])

            optimizer_u = torch.optim.Adam(netu.parameters(), lr=LR_FS)
            scheduler_u = torch.optim.lr_scheduler.ExponentialLR(optimizer_u, decreasing_FS)

            out_p = netp(inputs).reshape(-1, 1)
            p_abs_error = torch.abs(p_ex - out_p).reshape(-1, 1)
            p_l2 = torch.sum(p_abs_error ** 2 / (m * n))
            for in_epoch in range(in_iter):
                start2 = time.perf_counter()
                # train netu using netp
                optimizer_u.zero_grad()  # to make the gradients zero

                # boundary and initial conditions for netu
                u_train = np.hstack((u1_train.reshape(-1, 1), u2_train.reshape(-1, 1)))
                torch_X_train = Variable(torch.from_numpy(X_u_train).float(), requires_grad=False).to(device)
                torch_u_train = Variable(torch.from_numpy(u_train).float(), requires_grad=False).to(device)

                net_bc_out2 = netu(torch_X_train)
                mse_u = Loss(net_bc_out2, torch_u_train)

                # PDE loss for netu
                all_zeros = np.zeros((N_point, 2))
                pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)

                pde_u = loss_f(netu, netp, pt_x_collocation, pt_y_collocation, pt_t_collocation)
                mse_f = Loss(pde_u, pt_all_zeros)

                # total loss for network u
                loss2 = mse_u + mse_f
                loss2.backward()
                optimizer_u.step()
                scheduler_u.step()

                end2 = time.perf_counter()

                # L2-error of p and u  at every iteration
                output = netu(inputs)
                out_u1 = output[:, 0].reshape(-1, 1)
                out_u2 = output[:, 1].reshape(-1, 1)
                u1_abs_error = torch.abs(u1_ex - out_u1).reshape(-1, 1)
                u2_abs_error = torch.abs(u2_ex - out_u2).reshape(-1, 1)
                u1_l2 = torch.sum(u1_abs_error ** 2 / (m * n))
                u2_l2 = torch.sum(u2_abs_error ** 2 / (m * n))
                l2_FS = p_l2 + u1_l2 + u2_l2
                Time2 = end2 - start2
                Time_FS2_Sum = Time_FS2_Sum + Time2
                Time_Sum = Time_Sum + Time2
                Time_FS_l2_list.append([Time_Sum, p_l2.item(), u1_l2.item(), u2_l2.item(), l2_FS.item()])

            # updating netp_old and  netu_old
            netp_old = deepcopy(netp)
            netu_old = deepcopy(netu)

            # PINN
            if iter_list == vary_iter_list[0]:

                for in_epoch in range(in_iter):
                    start3 = time.perf_counter()
                    # training netn
                    optimizer_n.zero_grad()  # to make the gradients zero

                    # boundary and initial conditions for netn
                    u_train = np.hstack((u1_train.reshape(-1, 1), u2_train.reshape(-1, 1), p_train.reshape(-1, 1)))
                    torch_X_train = Variable(torch.from_numpy(X_u_train).float(), requires_grad=False).to(device)
                    torch_u_train = Variable(torch.from_numpy(u_train).float(), requires_grad=False).to(device)

                    net_bc_out3 = netn(torch_X_train)
                    mse_n1 = Loss(net_bc_out3, torch_u_train)

                    # PDE loss for netp
                    all_zeros = np.zeros((N_point, 3))
                    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)

                    pde_n = loss_N(netn, pt_x_collocation, pt_y_collocation, pt_t_collocation)
                    mse_n2 = Loss(pde_n, pt_all_zeros)

                    # total loss for network p
                    lossn = mse_n1 + mse_n2
                    lossn.backward()
                    optimizer_n.step()
                    scheduler_n.step()

                    end3 = time.perf_counter()

                    # L2-error of p and u  at every iteration
                    output = netn(inputs)
                    out_u1 = output[:, 0].reshape(-1, 1)
                    out_u2 = output[:, 1].reshape(-1, 1)
                    out_p = output[:, 2].reshape(-1, 1)
                    p_abs_error = torch.abs(p_ex - out_p).reshape(-1, 1)
                    u1_abs_error = torch.abs(u1_ex - out_u1).reshape(-1, 1)
                    u2_abs_error = torch.abs(u2_ex - out_u2).reshape(-1, 1)
                    p_l2 = torch.sum(p_abs_error ** 2 / (m * n))
                    u1_l2 = torch.sum(u1_abs_error ** 2 / (m * n))
                    u2_l2 = torch.sum(u2_abs_error ** 2 / (m * n))
                    l2 = p_l2 + u1_l2 + u2_l2
                    Time3 = end3 - start3
                    Time_PINN_Sum = Time_PINN_Sum + Time3
                    Time_Sum_PINN = Time_Sum_PINN + Time3
                    Time_PINN_l2_list.append([Time_Sum_PINN, p_l2.item(), u1_l2.item(), u2_l2.item(), l2.item()])

                print('iteration:', epoch, " Loss1:", loss1.item(), " Loss2:", loss2.item(), " Lossn:",
                      lossn.item(), " Time-FS:", Time_FS1_Sum + Time_FS2_Sum, " Time-PINN:", Time_PINN_Sum)

            else:
                print('iteration:', epoch, " Loss1:", loss1.item(), " Loss2:", loss2.item(), " Time-FS:", Time_FS1_Sum + Time_FS2_Sum)

    print("Total Training Time-FS:", Time_Sum, "Total Training Time-PINN:", Time_Sum_PINN, '\n')

    Train_Time.append([Time_Sum, Time_Sum_PINN])
    Time_FS_l2_list = np.array(Time_FS_l2_list).reshape(-1, 5)

    Time_PINN_l2_list = np.array(Time_PINN_l2_list).reshape(-1, 5)
    L2_Time_FS.append(Time_FS_l2_list)

print("Total Training Time:  FS-PINN,    PINN")
for time in Train_Time:
    print(*time)

"""
5. Visualization
"""
#  training time  vs  L2-error
legend_list = ['Classical PINN'] + ['FS-PINN with {0}'.format(*vary_iter_list[i]) for i in range(len(vary_iter_list))]
# p
fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy(Time_PINN_l2_list[:, 0], Time_PINN_l2_list[:, 1], linewidth='0.7', alpha=0.5)
for i in range(len(L2_Time_FS)):
    l2_list = L2_Time_FS[i]
    ax.semilogy(l2_list[:, 0], l2_list[:, 1], linewidth='0.7', alpha=0.5)
plt.legend(legend_list)
plt.xlabel('Time (s)')
plt.ylabel('MSE for pressure $p$')

# u1
fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy(Time_PINN_l2_list[:, 0], Time_PINN_l2_list[:, 2], linewidth='0.7', alpha=0.5)
for i in range(len(L2_Time_FS)):
    l2_list = L2_Time_FS[i]
    ax.semilogy(l2_list[:, 0], l2_list[:, 2], linewidth='0.7', alpha=0.5)
plt.legend(legend_list)
plt.xlabel('Time (s)')
plt.ylabel('MSE for x-displacement $u_1$')

# u2
fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy(Time_PINN_l2_list[:, 0], Time_PINN_l2_list[:, 3], linewidth='0.7', alpha=0.5)
for i in range(len(L2_Time_FS)):
    l2_list = L2_Time_FS[i]
    ax.semilogy(l2_list[:, 0], l2_list[:, 3], linewidth='0.7', alpha=0.5)
plt.legend(legend_list)
plt.xlabel('Time (s)')
plt.ylabel('MSE for y-displacement $u_2$')

# Total
fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy(Time_PINN_l2_list[:, 0], Time_PINN_l2_list[:, 4], linewidth='0.7', alpha=0.5)
for i in range(len(L2_Time_FS)):
    l2_list = L2_Time_FS[i]
    ax.semilogy(l2_list[:, 0], l2_list[:, 4], linewidth='0.7', alpha=0.5)
plt.legend(legend_list)
plt.xlabel('Time (s)')
plt.ylabel('Total MSE')

plt.show()
