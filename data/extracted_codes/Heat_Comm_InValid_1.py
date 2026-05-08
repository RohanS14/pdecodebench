import numpy as np
from scipy.integrate import odeint
from scipy.linalg import toeplitz

k = 1

# Spatial domain
n = 100
L = 1
dx = L / n
x = np.linspace(0, L, n)

T0 = 0
# Boundary conditions
Tn = 0.1
t_steps = 100
t = np.linspace(T0, Tn, t_steps)

U0 = 1
Un = -1

Uinit = np.zeros(shape=(n,))

A = toeplitz([-2.0, 1.0] + [0.0]*(n-2))

# Constructing the toeplitz matrix that forms the basis of our integrator
b = np.zeros(shape=(n))
b[0] = U0
b[n-1] = Un

def fn(u, t, A, b, k, dx):
    return ((k/dx**2)*np.matmul(-A, u) + (k/dx**3)*b)

# Performing the finite difference method usinng the Neumann Boundary Conditions
u = odeint(fn, Uinit, t, args=(A, b, k, dx))
# Solves using the scipy method odeint, which is a wrapper for the method LSODA from ODEPACK Fortran Library
