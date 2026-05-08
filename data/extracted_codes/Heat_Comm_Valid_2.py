import numpy as np
from scipy.integrate import odeint
from scipy.linalg import toeplitz

alpha = 0.0012
beta = -0.0015
Uc = 300

# Spacial domain
x0 = 0.0
xn = 1.0
n = 100
dx = (xn - x0)/n
x = np.linspace(x0, xn, n)

# Time steps
t0 = 0.0
tn = 300.0
t_steps = 300
dt = (tn - t0)/t_steps
t = np.linspace(t0, tn, t_steps)

# Boundary conditions
B0 = 325
B1 = 10

# Initial condition, T(x,0)=325 for all x in [0,1]
Uinit = np.ones(shape=(n,)) * 325

# Discretization matrix A
A = toeplitz([(-2*alpha / dx**2 + beta), (alpha / dx**2)] + [0.0]*(n-2))
A[n-1,n-1] = -alpha / dx**2 + beta

b = np.ones(shape=(n)) * -beta * Uc

b[0] = -beta * Uc + alpha / dx**2 * B0
b[n-1] = -beta * Uc + B1/dx

def fn(u, t, A, b):
    return (np.matmul(A, u) + b)

u = odeint(fn, Uinit, t, args=(A, b))
