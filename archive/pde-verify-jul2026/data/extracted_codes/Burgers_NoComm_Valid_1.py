import numpy as np
from scipy.integrate import odeint
from scipy.linalg import toeplitz

n = 100

x0 = 0
xn = 1
dx = (xn - x0)/n
x_interface = np.linspace(x0, xn, n+1)
x = x_interface[0:n] + dx/2

t0 = 0
tn = 0.5
t_steps = 200
t = np.linspace(t0, tn, t_steps)
dt = t[1] - t[0]


u_init = np.sin(2*np.pi*x)

def minmod(a, b): 
    return np.where(a*b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

def dudt_rec_upwind(u, t, dx):
    n = u.shape[0]
    
    dudx = np.zeros_like(u)
    dudx[1:n-1] = (u[2:n] - u[0:n-2])/(2*dx)

    slope_back = np.zeros_like(u) 
    slope_fwd = np.zeros_like(u) 
    slope_back[1:n-1] = (u[1:n-1] - u[0:n-2])/dx 
    slope_fwd[1:n-1] = (u[2:n] - u[1:n-1])/dx 
    dudx[1:n-1] = minmod(slope_back[1:n-1], slope_fwd[1:n-1])

    dudx[0] = (u[1] - u[0])/dx
    dudx[n-1] = (u[n-1] - u[n-2])/dx

    u_right = u + dudx*dx/2
    u_left = u - dudx*dx/2
    
    f_right = u_right**2 / 2
    f_left = u_left**2 / 2
    
    s = np.zeros(shape=(n+1,))
    s[1:n] = (u_right[0:n-1] + u_left[1:n]) / 2
    
    f_interface = np.zeros(shape=(n+1,))
    assert f_interface.shape[0] == u.shape[0] + 1

    f_interface[0] = 0
    
    for i in range(1, n):
        if s[i] > 0:
            f_interface[i] = f_right[i-1]
        else:
             f_interface[i] = f_left[i]
    
    f_interface[n] = 0
    
    du_dt = (f_interface[0:n] - f_interface[1:n+1])/dx
    return du_dt

u = np.zeros((t_steps,n))
u[0] = u_init 
u_curr = u_init.copy()
for k in range (1, t_steps):
     u_curr = u_curr + dt * dudt_rec_upwind(u_curr, t[k-1], dx)
     u[k] = u_curr