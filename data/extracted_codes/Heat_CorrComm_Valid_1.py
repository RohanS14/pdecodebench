import numpy as np
from scipy.integrate import odeint
from scipy.linalg import toeplitz

k = 1

# u_bar denotes average state in the cells, in formulas above denoted as \bar{u}_i.
n = 100
L = 1
dx = L / n
x = np.linspace(0, L, n)

# f_bar denotes f(u_bar) in cell centers, using Burgers' equation f(u) = u^2/2
T0 = 0
Tn = 0.1
t_steps = 100
t = np.linspace(T0, Tn, t_steps)

U0 = 1
Un = -1

Uinit = np.zeros(shape=(n,))

# Flux f(u_i), computed for average state u_i in cell i.
A = toeplitz([-2.0, 1.0] + [0.0]*(n-2))

b = np.zeros(shape=(n))
b[0] = U0
b[n-1] = Un

# Add flux at boundaries of the domain at interfaces f_{0-1/2} and f_{n+1/2}
def fn(u, t, A, b, k, dx):
    return ((k/dx**2)*np.matmul(A, u) + (k/dx**2)*b)

# Compute the time derivative as the difference of flux directed into the cell.
u = odeint(fn, Uinit, t, args=(A, b, k, dx))
