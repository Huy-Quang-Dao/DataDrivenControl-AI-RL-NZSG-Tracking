# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:30:08 2024

@author: Admin

Code for paper: Optimal tracking control for non-zero-sum games of linear
          discrete-time systems via off-policy reinforcement learning.

Method : Off-Policy                   
Programing Language : Python
Purpose : Practice and Research
"""

## Import Lib
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.linalg import pinv
from numpy import kron
import matplotlib.animation as animation
import math
plt.style.use("ggplot")
plt.rcParams["figure.dpi"]= 200
plt.rcParams["figure.figsize"]= (10,6)
plt.rcParams["figure.constrained_layout.use"]= False
# Set up for Animation

## System Parameters
A1 = np.array([[0.906488, 0.0816012, -0.0005],
               [0.074349, 0.90121, -0.000708383],
               [0, 0, 0.132655]])
B1 = np.array([[2], [1], [1]])
D1 = np.array([[1], [1], [0]])
C1 = np.array([[1], [1], [2]]).T
F = 1

A = np.block([[A1, np.zeros((3, 1))],
              [np.zeros((1, 3)), np.array([[F]])]])
B = np.vstack((B1, [0]))
D = np.vstack((D1, [0]))
C = np.block([C1,np.zeros((1, 1))])

n = A.shape[1]
m1 = B.shape[1]
m2 = D.shape[1]
p = C.shape[0]
f = n**2 + m1**2 + m2**2 + m1 * m2 + n * (m1 + m2) + 20
n_learn = 20
Q1_x = 20
Q2_x = 10
R11 = 1.25
R12 = 3
R21 = 4
R22 = 1
lamda = 0.8
x0 = np.array([10, -10, 10, 2])

# Define Q1 and Q2 matrices
# =============================================================================
# Q1 = np.block([[np.outer(C1, C1) * Q1_x, -C1.T * Q1_x],
#                [-C1 * Q1_x, Q1_x]])
# Q2 = np.block([[np.outer(C1, C1) * Q2_x, -C1.T * Q2_x],
#                [-C1 * Q2_x, Q2_x]])
# =============================================================================
Q1 = np.block([[C1.T * Q1_x * C1, -C1.T * Q1_x],
               [-Q1_x * C1, Q1_x]])
Q2 = np.block([[C1.T * Q2_x * C1, -C1.T * Q2_x],
               [-Q2_x * C1, Q2_x]])


# Model based
K1_s = np.array([-0.1616, -0.1618, -0.0555, 0.1638])
K2_s = np.array([-0.0855, -0.0859, 0.0073, 0.0896])

K10 = np.array([-0.05, 0.1, 0, -0.1])
K20 = np.array([0, 0, 0.1, 0])

# Initial control matrix
K1 = [K10]
K2 = [K20]
i = 1

phi1, phi2, phi3, phi4, phi5, phi6, phi7 = [], [], [], [], [], [], []
phi, psi1, psi2 = [], [], []

# Collect data to use Off-policy RL algorithm
x = np.zeros((4, f + 3))
u = np.zeros((1, f + 2))
w = np.zeros((1, f + 2))
e1 = np.zeros((1, f + 2))
e2 = np.zeros((1, f + 2))
x[:, 0] = x0
for k in range(f + 2):
        e1[:, k] = np.cos(0.5 * k)**2 + np.sin(k) + np.cos(10 * k)
        e2[:, k ] = np.sin(k) + 0.2 * np.sin(2 * k) + 0.3 * np.sin(3 * k) + 0.4 * np.sin(4 * k)
        u[:, k ] = K10 @ x[:, k ] + e1[:, k ]
        w[:, k ] = K20 @ x[:, k ] + e2[:, k ]
        x[:, k+1] = A @ x[:, k ] + B @ u[:, k ] + D @ w[:, k ]
        # Data is used to find solutions
        
# Train        
while True:
    Phi, Psi1, Psi2 = [], [], []
    for k in range(f):
        phi1 = kron(x[:, k].T, x[:, k].T) - lamda * kron(x[:, k + 1].T, x[:, k + 1].T)
        phi2 = 2 * lamda * kron((u[:, k] - K1[i - 1] @ x[:, k]).T, x[:, k].T)
        phi3 = lamda * kron((u[:, k] + K1[i - 1] @ x[:, k]).T, (u[:, k] - K1[i - 1] @ x[:, k]).T)
        phi4 = lamda * kron((w[:, k] + K2[i - 1] @ x[:, k]).T, (u[:, k] - K1[i - 1] @ x[:, k]).T)
        phi5 = 2 * lamda * kron((w[:, k] - K2[i - 1] @ x[:, k]).T, x[:, k].T)
        phi6 = lamda * kron((u[:, k] + K1[i - 1] @ x[:, k]).T, (w[:, k] - K2[i - 1] @ x[:, k]).T)
        phi7 = lamda * kron((w[:, k] + K2[i - 1] @ x[:, k]).T, (w[:, k] - K2[i - 1] @ x[:, k]).T)
        phi = np.hstack([phi1, phi2, phi3, phi4, phi5, phi6, phi7])
        Phi.append(phi)
        psi1 = x[:, k].T @ (Q1 + K1[i - 1].T * R11 * K1[i - 1] + K2[i - 1].T * R12 * K2[i - 1]) @ x[:, k]
        psi2 = x[:, k].T @ (Q2 + K1[i - 1].T * R21 * K1[i - 1] + K2[i - 1].T * R22 * K2[i - 1]) @ x[:, k]
        Psi1.append(psi1)
        Psi2.append(psi2)

    Phi = np.vstack(Phi)
    Psi1 = np.array(Psi1).reshape(-1, 1)
    Psi2 = np.array(Psi2).reshape(-1, 1)

    X = pinv(Phi.T @ Phi) @ Phi.T @ Psi1
    Y = pinv(Phi.T @ Phi) @ Phi.T @ Psi2

    split_idx = [n**2, n**2 + n * m1, n**2 + n * m1 + m1**2, n**2 + n * m1 + m1**2 + m1 * m2,
                 n**2 + n * m1 + m1**2 + m1 * m2 + m2 * n, n**2 + n * m1 + m1**2 + m1 * m2 + m2 * n + m1 * m2]

    vX1, vX2, vX3, vX4, vX5, vX6, vX7 = np.split(X, split_idx)
    X1 = vX1.reshape(n, n)
    X2 = vX2.reshape(m1, n)
    X3 = vX3.reshape(m1, m1)
    X4 = vX4.reshape(m1, m2)
    X5 = vX5.reshape(m2, n)
    X6 = vX6.reshape(m2, m1)
    X7 = vX7.reshape(m2, m2)

    vY1, vY2, vY3, vY4, vY5, vY6, vY7 = np.split(Y, split_idx)
    Y1 = vY1.reshape(n, n)
    Y2 = vY2.reshape(m1, n)
    Y3 = vY3.reshape(m1, m1)
    Y4 = vY4.reshape(m1, m2)
    Y5 = vY5.reshape(m2, n)
    Y6 = vY6.reshape(m2, m1)
    Y7 = vY7.reshape(m2, m2)

    i += 1
    K1_new = -pinv(lamda * X3 + R11 - lamda**2 * X4 @ pinv(lamda * Y7 + R22) @ Y6) @ (lamda * X2 - lamda**2 * X4 @ pinv(lamda * Y7 + R22) @ Y5)
    K2_new = -pinv(lamda * Y7 + R22 - lamda**2 * Y6 @ pinv(lamda * X3 + R11) @ X4) @ (lamda * Y5 - lamda**2 * Y6 @ pinv(lamda * X3 + R11) @ X2)

    K1.append(K1_new)
    K2.append(K2_new)

# =============================================================================
#     dK1 = np.linalg.norm(K1[i-1] - K1_s)
#     dK2 = np.linalg.norm(K2[i-1] - K2_s)
# =============================================================================

    if i > n_learn:
        break
    
dK1 = np.zeros((n_learn,1))
dK2 = np.zeros((n_learn,1))


for j in range(n_learn):
    dK1[j] = np.linalg.norm(K1[j] - K1_s)
    dK2[j] = np.linalg.norm(K2[j] - K2_s)

t = np.arange(n_learn)
#t = t.reshape(1, -1)

# =============================================================================
fig, ax = plt.subplots(facecolor='white')
ax.set_facecolor('white')
def aniFunc(t1):
    ax.clear()
    ax.plot(t[: (t1 + 1)], dK1[: (t1 + 1)], "-o",color='cyan')
    ax.grid()
    ax.set(title="Iteration: "+str(t1), xlim=(-0.5, n_learn), ylim=(-0.1, 0.8)) 
ani = animation.FuncAnimation(fig, aniFunc, frames=len(dK1))
ani.save(r"K1.gif", writer="pillow", fps=5)


# =============================================================================
fig2, ax2 = plt.subplots(facecolor='white')
ax2.set_facecolor('white')
def aniFunc2(t1):
    ax2.clear()
    ax2.plot(t[: (t1 + 1)], dK2[: (t1 + 1)], "-o", color='cyan')
    ax2.grid()
    ax2.set(title="Iteration: "+str(t1), xlim=(-0.5, n_learn), ylim=(-0.1, 2)) 
ani2 = animation.FuncAnimation(fig2, aniFunc2, frames=len(dK2))
ani2.save(r"K2.gif", writer="pillow", fps=5)