from time import time
import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
nu = 0.000625
T = 0.1
dt = 0.01
M = 6
N = [2**M, 2**M, 2**M]
L = np.array([2*np.pi, 4*np.pi, 4*np.pi], dtype=float) 
FFT = PFFT(MPI.COMM_WORLD, N, collapse=False)
FFT_pad = FFT
# U is velocity in physical space, U_hat is velocity in fourier space 
U = newDistArray(FFT, False, rank=1, view=True)      
U_hat = newDistArray(FFT, rank=1, view=True)       
P = newDistArray(FFT, False, view=True)               
P_hat = newDistArray(FFT, view=True)                 
U_hat0 = newDistArray(FFT, rank=1, view=True)         
U_hat1 = newDistArray(FFT, rank=1, view=True) 
# RK4 coefficients for explicit time integration        
a = [1./6., 1./3., 1./3., 1./6.]        
b = [0.5, 0.5, 1.]                     
dU = newDistArray(FFT, rank=1, view=True)            
curl = newDistArray(FFT, False, rank=1, view=True)
U_pad = newDistArray(FFT_pad, False, rank=1, view=True)
curl_pad = newDistArray(FFT_pad, False, rank=1, view=True)
def get_local_mesh(FFT, L):
    # local spatial grid 
    X = np.ogrid[FFT.local_slice(False)]
    N = FFT.global_shape()
    X = [np.broadcast_to(x*L[i]/N[i], FFT.shape(False)) for i, x in enumerate(X)]
    return X
def get_local_wavenumbermesh(FFT, L):
    # fourier wavenumbers
    s = FFT.local_slice()
    N = FFT.global_shape()
    k = [np.fft.fftfreq(n, 1./n).astype(int) for n in N[:-1]]
    k.append(np.fft.rfftfreq(N[-1], 1./N[-1]).astype(int))
    K = [ki[si] for ki, si in zip(k, s)]
    Ks = np.meshgrid(*K, indexing='ij', sparse=True)
    Lp = 2*np.pi/L
    return [np.broadcast_to(k*Lp[i], FFT.shape(True)) for i, k in enumerate(Ks)]
X = get_local_mesh(FFT, L)
K = get_local_wavenumbermesh(FFT, L)
K = np.array(K).astype(float)
K2 = np.sum(K*K, 0, dtype=float)
K_over_K2 = K.astype(float) / K2.astype(float)
def cross(x, y, z):
    # nonlinear term, computing in physical space then transforming back to fourier space 
    z[0] = FFT_pad.forward(x[1]*y[2]-x[2]*y[1], z[0])
    z[1] = FFT_pad.forward(x[2]*y[0]-x[0]*y[2], z[1])
    z[2] = FFT_pad.forward(x[0]*y[1]-x[1]*y[0], z[2])
    return z
def compute_curl(x, z):
    # compute vorticity using spectral derivatives 
    z[2] = FFT_pad.backward(1j*(K[0]*x[1]-K[1]*x[0]), z[2])
    z[1] = FFT_pad.backward(1j*(K[2]*x[0]-K[0]*x[2]), z[1])
    z[0] = FFT_pad.backward(1j*(K[1]*x[2]-K[2]*x[1]), z[0])
    return z
def compute_rhs(rhs):
    for j in range(3):
        U_pad[j] = FFT_pad.backward(U_hat[j], U_pad[j])
    curl_pad[:] = compute_curl(U_hat, curl_pad)
    rhs = cross(U_pad, curl_pad, rhs)
    # Spectral projection 
    P_hat[:] = np.sum(rhs*K_over_K2, 0, out=P_hat)
    rhs -= P_hat*K
    # Adding viscosity 
    rhs -= nu*K2*U_hat
    return rhs
U[0] = np.sin(X[0])*np.cos(X[1])*np.cos(X[2])
U[1] = -np.cos(X[0])*np.sin(X[1])*np.cos(X[2])
U[2] = 0
for i in range(3):
    U_hat[i] = FFT.forward(U[i], U_hat[i])
t = 0.0
tstep = 0
t0 = time()
while t < T-1e-8:
    t += dt
    tstep += 1
    U_hat1[:] = U_hat0[:] = U_hat
    # explicit rk4 time stepping 
    for rk in range(4):
        dU = compute_rhs(dU)
        if rk < 3:
            U_hat[:] = U_hat0 + b[rk]*dt*dU
        U_hat1[:] += a[rk]*dt*dU
    U_hat[:] = U_hat1[:]
    for i in range(3):
        U[i] = FFT.backward(U_hat[i], U[i])
k = MPI.COMM_WORLD.reduce(np.sum(U*U)/N[0]/N[1]/N[2]/2)
if MPI.COMM_WORLD.Get_rank() == 0:
    print('Time = {}'.format(time()-t0))
FFT.destroy()