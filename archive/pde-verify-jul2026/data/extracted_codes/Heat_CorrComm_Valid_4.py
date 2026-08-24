import numpy as np



k = 0.001



n = 50

L = 1

dx = L/n

x = np.linspace(0, L, n)



T0 = 0

Tn = 240

t_steps = (Tn - T0)*10

dt = (Tn - T0)/t_steps



# Burgers flux

U0 = 100

U1 = 0



# Minimum flux at u*

Uinit = 0



# u denotes state in cell centers (in the formulas sometimes written as \bar{u})

CFL = k * dt / dx**2

print(f"CFL: {CFL:0.3f} < 0.5")

assert CFL < 0.5



u = np.ones(shape=(t_steps, n)) * Uinit



# f denotes f(u) in cell centers, using Burgers' equation f(u) = u^2/2

D = np.zeros(shape=(n, n))

for i in range(1, n-1):

    D[i, i-1] = 1

    D[i, i] = -2

    D[i, i+1] = 1



# Reconstruct u_i_plus_half_left

dudt = np.zeros(shape=(n))



for t in range(0, t_steps-1):

    u[t, 0] = U0

    u[t, n-1] = U1

    dudt = k * np.matmul(D, u[t])/dx**2

    u[t+1] = u[t] + dudt * dt
