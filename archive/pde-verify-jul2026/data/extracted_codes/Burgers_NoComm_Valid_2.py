import numpy as np
from scipy.integrate import odeint

n = 100
u = np.zeros(shape=(n,))

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

def limiter(r):
    _r = r.copy()
    _r[_r < 0] = 0
    phi = 2*_r/(1 + _r)
    phi[~np.isfinite(_r)] = 0
    return phi


def godunov(ul, ur):
    fL = ul**2 / 2
    fR = ur**2 / 2
    f_min = 0
    f = np.zeros_like(ul)

    f[ul > ur] = np.maximum(fL[ul > ur], fR[ul > ur])

    f[ul <= ur] = np.minimum(fL[ul <= ur], fR[ul <= ur])
    f[np.where((ul <= 0) & (ur > 0))] = f_min
    return f

def dudt(u, t, dx):
    n = len(u)

    u_im1 = np.hstack([0, u[0: n-1]])
    u_ip1 = np.hstack([u[1:n], 0])
    u_ip2 = np.hstack([u_ip1[1:n], 0])

    assert len(u) == n
    assert len(u_im1) == n
    assert len(u_ip1) == n
    assert len(u_ip2) == n
    assert u_im1[1] == u[0]
    assert u_ip1[1] == u[2]
    assert u_ip2[1] == u[3]
    
    r = (u_ip1 - u)/(u - u_im1)
    phi = limiter(r)
    u_iph_left = u + 0.5*(u - u_im1)*phi
    
    r = (u - u_ip1)/(u_ip1 - u_ip2)
    phi = limiter(r)
    u_iph_right = u_ip1 + 0.5*(u_ip1 - u_ip2)*phi
    
    f_interface = godunov(u_iph_left, u_iph_right)
    f_interface = np.hstack([0, f_interface])

    du_dt = (f_interface[0:n] - f_interface[1:n+1])/dx
    return du_dt

u = np.zeros((t_steps,n))
u[0] = u_init 
u_curr = u_init.copy()
for k in range (1, t_steps):
     u_curr = u_curr + dt * dudt(u_curr, t[k-1], dx)
     u[k] = u_curr