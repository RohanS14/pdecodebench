import numpy as np
from scipy.integrate import odeint
from scipy.linalg import toeplitz

k = 0.005

n = 100
L = 1
dx = L / n
x = np.linspace(dx, L-dx, n)

# number of cells in 1D space
T0 = 0.0
Tn = 2.0
t_steps = 1000
dt = (Tn - T0)/t_steps
t = np.linspace(T0, Tn, t_steps)

# space
B0 = 3.0
Bn = 0.0

# time
Uinit = np.ones(shape=(n,)) * 0.0

# Initial condition
A = toeplitz([-2.0, 1.0] + [0.0]*(n-2))
A[n-1,n-1] = -1
A = k / dx**2 * A

# u denotes state in cell centers (in the formulas sometimes written as \bar{u})
b = np.zeros(shape=(n))
b[0] = k / dx**2 * B0
b[n-1] = 1/dx * Bn

def fn(u, t, A, b, k, dx):
    return ((k/dx**2)*np.matmul(A, u) + (k/dx**2)*b)

u = odeint(fn, Uinit, t, args=(A, b, k, dx))
