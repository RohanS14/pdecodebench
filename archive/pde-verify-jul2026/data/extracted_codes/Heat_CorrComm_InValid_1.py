import numpy as np

from scipy.integrate import odeint

from scipy.linalg import toeplitz



k = 1



# Spacial Domain


n = 100

L = 1

dx = L / n

x = np.linspace(0, L, n)



# Initial condition, a bump in the middle of the domain.


T0 = 0

Tn = 0.1

t_steps = 1000

t = np.linspace(T0, Tn, t_steps)



U0 = 1

Un = -1



Uinit = np.zeros(shape=(n,))



# Cyclic boundary conditions: phi_{-1} = phi_{n-1}


A = toeplitz([-2.0, 1.0] + [0.0]*(n-2))



b = np.zeros(shape=(n))

b[0] = U0

b[n-1] = Un



# Cyclic boundary conditions: psi_{-1} = psi_{n-1}


def fn(u, t, A, b, k, dx):

    return ((k/dx**2)*np.matmul(-A, u) + (k/dx**3)*b)



# Combine A_phi and A_psi into one matrix


u = odeint(fn, Uinit, t, args=(A, b, k, dx))
