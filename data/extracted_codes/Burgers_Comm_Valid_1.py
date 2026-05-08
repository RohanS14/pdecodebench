import numpy as np
from scipy.integrate import odeint
from scipy.linalg import toeplitz

# number of cells in 1D space
n = 100

# space
x0 = 0
xn = 1
dx = (xn - x0)/n
x_interface = np.linspace(x0, xn, n+1)
x = x_interface[0:n] + dx/2

# time
t0 = 0
tn = 0.5
t_steps = 200
t = np.linspace(t0, tn, t_steps)


# Initial condition
u_init = np.sin(2*np.pi*x)

def dudt_rec_upwind(u, t, dx):
    # u denotes state in cell centers (in the formulas sometimes written as \bar{u})
    n = u.shape[0]
    
    dudx = np.zeros_like(u)
    dudx[1:n-1] = (u[2:n] - u[0:n-2])/(2*dx)
    dudx[0] = (u[1] - u[0])/dx
    dudx[n-1] = (u[n-1] - u[n-2])/dx

    u_right = u + dudx*dx/2
    u_left = u - dudx*dx/2
    
    # Compute the flux for every cell i, not in the middle, but at every cell interface _inside_ each cell.
    f_right = u_right**2 / 2
    f_left = u_left**2 / 2
    
    # Shock speed at the cell interfaces from f_{0-1/2} to f_{n+1/2}, 
    s = np.zeros(shape=(n+1,))
    s[1:n] = (u_right[0:n-1] + u_left[1:n]) / 2
    
    # Flux at the cell interfaces from f_{0-1/2} to f_{n+1/2}
    f_interface = np.zeros(shape=(n+1,))
    assert f_interface.shape[0] == u.shape[0] + 1

    f_interface[0] = 0
    
    for i in range(1, n):
        if s[i] > 0:
            f_interface[i] = f_right[i-1]
        else:
             f_interface[i] = f_left[i]
    
    f_interface[n] = 0
    
    # Compute the time derivative as the difference of flux directed into the cell.
    du_dt = (f_interface[0:n] - f_interface[1:n+1])/dx
    return du_dt

u = odeint(dudt_rec_upwind, u_init, t, args=(dx,))
