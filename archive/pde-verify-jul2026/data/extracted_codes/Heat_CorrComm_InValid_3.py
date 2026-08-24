import numpy as np

from scipy.integrate import odeint

from scipy.linalg import toeplitz



# number of cells in 1D space


k = 1.0



# space


n = 100

L = 1

dx = L / n

x = np.linspace(0, L, n)



# time


T0 = 0.0

Tn = 0.1

t_steps = 100

t = np.linspace(T0, Tn, t_steps)



# Initial condition


B0 = 1.0

Bn = -1.0



# returns 0 if a,b have opposite signs (or either is 0); else smaller one


Uinit = np.zeros(shape=(n,))



# u denotes state in cell centers (in the formulas sometimes written as \bar{u})


A = toeplitz([-2.0, -1.0] + [0.0]*(n-2))



# Compute the flux for every cell i, not in the middle, but at every cell interface _inside_ each cell.


b = np.zeros(shape=(n))

b[0] = B0

b[n-1] = Bn



def fn(u, t, A, b, k, dx):

    return ((k/dx**2)*np.matmul(A, u) + (k/dx**2)*b)



u = odeint(fn, Uinit, t, args=(A, b, k, dx))
