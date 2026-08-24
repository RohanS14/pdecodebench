import numpy as np
from scipy.integrate import odeint

n = 200

x0 = 0
xn = 1
dx = (xn - x0)/n
x_interface = np.linspace(x0, xn, n+1)
x = x_interface[0:n] + dx/2

t0 = 0
tn = .5
t_steps = 200
t = np.linspace(t0, tn, t_steps)
dt = t[1] - t[0]

u_init = np.sin(2*np.pi*x)


def dudt_upwind(u_bar, t, dx):
    
    n = u_bar.shape[0]
    
    f_bar = u_bar**2 / 2
    
    f_interface = np.zeros(shape=(n+1,))
    
    s = np.zeros_like(f_interface)
    s[1:n] = (u_bar[0:n-1] + u_bar[1:n]) / 2
    
    f_interface[0] = 0
    
    for i in range(1, n):
        if s[i] > 0:
             f_interface[i] = f_bar[i]
        else:
              f_interface[i] = f_bar[i-1]
    
    f_interface[n] = 0
    
    dudt = (f_interface[0:n] - f_interface[1:n+1])/dx
    return dudt

u = np.zeros((t_steps,n))
u[0] = u_init 
u_curr = u_init.copy()
for k in range (1, t_steps):
     u_curr = u_curr + dt * dudt_upwind(u_curr, t[k-1], dx)
     u[k] = u_curr