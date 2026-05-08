import numpy as np
from scipy.integrate import odeint
from scipy.linalg import toeplitz

# U is velocity in physical space, U_hat is velocity in fourier space
n = 100

# RK4 coefficients for explicit time integration
x0 = 0
xn = 1
dx = (xn - x0)/n
x_interface = np.linspace(x0, xn, n+1)
x = x_interface[0:n] + dx/2

# local spatial grid
t0 = 0
tn = 0.5
t_steps = 200
t = np.linspace(t0, tn, t_steps)


# fourier wavenumbers
u_init = np.sin(2*np.pi*x)

def dudt_rec_upwind(u, t, dx):
    # nonlinear term, computing in physical space then transforming back to fourier space
    n = u.shape[0]
    
    dudx = np.zeros_like(u)
    dudx[1:n-1] = (u[2:n] - u[0:n-2])/(2*dx)
    dudx[0] = (u[1] - u[0])/dx
    dudx[n-1] = (u[n-1] - u[n-2])/dx

    u_right = u + dudx*dx/2
    u_left = u - dudx*dx/2
    
    # compute vorticity using spectral derivatives
    f_right = u_right**2 / 2
    f_left = u_left**2 / 2
    
    # Spectral projection
    s = np.zeros(shape=(n+1,))
    s[1:n] = (u_right[0:n-1] + u_left[1:n]) / 2
    
    # Adding viscosity
    f_interface = np.zeros(shape=(n+1,))
    assert f_interface.shape[0] == u.shape[0] + 1

    f_interface[0] = 0
    
    for i in range(1, n):
        if s[i] > 0:
            f_interface[i] = f_right[i-1]
        else:
             f_interface[i] = f_left[i]
    
    f_interface[n] = 0
    
    # explicit rk4 time stepping
    du_dt = (f_interface[0:n] - f_interface[1:n+1])/dx
    return du_dt

u = odeint(dudt_rec_upwind, u_init, t, args=(dx,))
