# -*- coding: utf-8 -*-
"""
@author: Huipeng Gu, Pengxiang Hong.
The benchmark test is a Dirichlet problem.

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

# recording list
L2_list = []

# weight option
weigh1 = [1, 1]
weigh2 = [1, 1]


# training
for epoch in range(1, out_iter + 1):

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

    # step1
    optimizer_p = torch.optim.Adam(netp.parameters(), lr=LR_FS)
    scheduler_p = torch.optim.lr_scheduler.ExponentialLR(optimizer_p, decreasing_FS)
    start1 = time.perf_counter()
    for i in range(1, in_iter + 1):
        # train netp using netp_old, netu_old
        optimizer_p.zero_grad()

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

    # step2
    optimizer_u = torch.optim.Adam(netu.parameters(), lr=LR_FS)
    scheduler_u = torch.optim.lr_scheduler.ExponentialLR(optimizer_u, decreasing_FS)
    start2 = time.perf_counter()
    for i in range(1, in_iter + 1):
        optimizer_u.zero_grad()

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

    # updating netp_old and  netu_old
    netp_old = deepcopy(netp)
    netu_old = deepcopy(netu)

    # recording
    Time1 = end1 - start1 + end2 - start2
    print('iteration:{0:5d}, loss1: {1:14.12f} loss2: {2:14.12f} Time:{3:14.12f} '
          .format(epoch, loss1,  loss2, Time1))

    # L2-error of p and u  at every iteration
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

    L2_list.append([epoch, p_l2.item(), u1_l2.item(), u2_l2.item()])

# MSE in every iteration
print('MSE for p,u1 and u2.')
for i in L2_list:
    print('iteration:{0:3d}, p:{1:8f}, u1:{2:8f}, u2:{3:8f}, Total:{4:8f}'.format(i[0], i[1], i[2], i[3],
                                                                                  i[1] + i[2] + i[3]))
"""
5. Visualization
"""
#  picture points
x = torch.linspace(0, 1, 500)
y = torch.linspace(0, 1, 500)
m, n = x.shape[0], y.shape[0]
X, Y = torch.meshgrid(x, y)
point = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
t = torch.ones((m * n, 1)) * T_final
inputs = torch.hstack((point, t)).to(device)
Xnp = X.detach().numpy()
Ynp = Y.detach().numpy()
T = T_final

outp = netp(inputs).cpu().detach().numpy()
outu = netu(inputs).cpu().detach().numpy()

p_fs = outp.reshape(m, n)
u1_fs = outu[:, 0].reshape(m, n)
u2_fs = outu[:, 1].reshape(m, n)

u1_ex = 0.5 * T * Xnp * Xnp
u2_ex = 0.5 * T * Ynp * Ynp
p_ex = np.sin(Xnp + Ynp) * np.exp(T)

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

outp = netp(inputs).cpu().detach().numpy()
outu = netu(inputs).cpu().detach().numpy()

p_fs_vector = outp.reshape(N, N)
u1_fs_vector = outu[:, 0].reshape(N, N)
u2_fs_vector = outu[:, 1].reshape(N, N)

u1_ex_vector = 0.5 * T * XX2 * XX2
u2_ex_vector = 0.5 * T * YY2 * YY2
p_ex_vector = np.sin(XX2 + YY2) * np.exp(T)


# figure
fig, ax = plt.subplots()
levels = np.arange(0, 1.8, 0.01)
cs = ax.contourf(Xnp, Ynp, p_fs)
plt.quiver(XX2, YY2, u1_fs_vector, u2_fs_vector, headwidth=3, alpha=1)
cs.cmap.set_under('blue')
cs.changed()
cbar = fig.colorbar(cs)
plt.title('FS-PINN solution')

fig, ax = plt.subplots()
levels = np.arange(0, 1.8, 0.01)
cs = ax.contourf(Xnp, Ynp, p_ex)
plt.quiver(XX2, YY2, u1_ex_vector, u2_ex_vector, headwidth=3, alpha=1)
cbar = fig.colorbar(cs)
plt.title('Analytical solution')

plt.show()

