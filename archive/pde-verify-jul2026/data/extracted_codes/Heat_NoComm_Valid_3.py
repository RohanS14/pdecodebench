import numpy as np

from scipy.integrate import odeint

from scipy.linalg import toeplitz





k = 1





n = 100

L = 1

dx = L / n

x = np.linspace(0, L, n)





T0 = 0

Tn = 0.1

t_steps = 100

t = np.linspace(T0, Tn, t_steps)





U0 = 1

Un = -1





Uinit = np.zeros(shape=(n,))





A = toeplitz([-2.0, 1.0] + [0.0]*(n-2))





b = np.zeros(shape=(n))

b[0] = U0

b[n-1] = Un



def fn(u, t, A, b, k, dx):

    return ((k/dx**2)*np.matmul(A, u) + (k/dx**2)*b)



u = odeint(fn, Uinit, t, args=(A, b, k, dx))
