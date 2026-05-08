import numpy as np
from scipy.integrate import odeint
from scipy.linalg import toeplitz

alpha = 0.0012
beta = -0.0015
Uc = 300

x0 = 0.0
xn = 1.0
n = 100
dx = (xn - x0)/n
x = np.linspace(x0, xn, n)

t0 = 0.0
tn = 300.0
t_steps = 300
dt = (tn - t0)/t_steps
t = np.linspace(t0, tn, t_steps)

B0 = 325
B1 = 10

Uinit = np.ones(shape=(n,)) * 325

A = toeplitz([(-2*alpha / dx**2 + beta), (alpha / dx**2)] + [0.0]*(n-2))
A[n-1,n-1] = -alpha / dx**2 + beta

b = np.ones(shape=(n)) * -beta * Uc

b[0] = -beta * Uc + alpha / dx**2 * B0
b[n-1] = -beta * Uc + B1/dx

def fn(u, t, A, b):
    return (np.matmul(A, u) + b)

u = odeint(fn, Uinit, t, args=(A, b))
