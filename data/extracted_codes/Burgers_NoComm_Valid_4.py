import numpy as np
from scipy.integrate import odeint

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

u_init = np.sin(2*np.pi*x)

def dudt_central(u_bar, t, dx):
    f_bar = u_bar**2 / 2

    
    f_interface = (f_bar[0:n-1] + f_bar[1:n]) / 2

    
    f_interface = np.hstack([0, f_interface, 0])

    
    assert len(f_interface) == len(u_bar) + 1

  
    du_dt = (f_interface[0:n] - f_interface[1:n+1])/dx
    return du_dt

u = odeint(dudt_central, u_init, t, args=(dx,))
