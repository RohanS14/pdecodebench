import numpy as np

from scipy.integrate import odeint

c = 1

# Burgers flux

n = 200

L = 2*np.pi

dx = L / n

x = np.linspace(0, L, n)



T0 = 0

Tn = 2*np.pi

t_steps = 200

t = np.linspace(T0, Tn, t_steps)



# Minimum flux at u*

phi_init = np.sin(x/2)**16

psi_init = 0*phi_init



phi_psi_init = np.hstack([phi_init, psi_init])



A_phi = np.zeros(n)

A_phi = A_phi + np.diag( 1/(2*dx)*np.ones(n-1,),  1)

A_phi = A_phi + np.diag(-1/(2*dx)*np.ones(n-1,), -1)



# u denotes state in cell centers (in the formulas sometimes written as \bar{u})

A_phi[0,n-1] = -1/(2*dx)

A_phi[n-1, 0] = 1/(2*dx)



A_psi = np.zeros(n)

A_psi = A_psi + np.diag( 1/(2*dx)*np.ones(n-1,),  1)

A_psi = A_psi + np.diag(-1/(2*dx)*np.ones(n-1,), -1)



# f denotes f(u) in cell centers, using Burgers' equation f(u) = u^2/2

A_psi[0,n-1] = -1/(2*dx)

A_psi[n-1, 0] = 1/(2*dx)



# Reconstruct u_i_plus_half_left

A = np.zeros(shape=(2*n,2*n))

A[:n, :n] = A_phi

A[n:, n:] = A_psi



def ddt(phi_psi, t, A, n):

    ddx =  np.matmul(A, phi_psi)

    dphidx = ddx[:n]

    dpsidx = ddx[n:]

    dpsidt = dphidx

    dphidt = dpsidx

    return np.hstack([dphidt, dpsidt])



phi_psi = odeint(ddt, phi_psi_init, t, args=(A, n))
