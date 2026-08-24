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



# Boundary conditions

U0 = 100

U1 = 0



# Initial condition, u(x,0)=0

Uinit = 0



# Courant-Friedrichs-Levy number

CFL = k * dt / dx**2

print(f"CFL: {CFL:0.3f} < 0.5")

assert CFL < 0.5



u = np.ones(shape=(t_steps, n)) * Uinit



# Differential matrix. (Note that the matrix includes the nodes at the walls. The derivative is zero at the walls)

D = np.zeros(shape=(n, n))

for i in range(1, n-1):

    D[i, i-1] = 1

    D[i, i] = -2

    D[i, i+1] = 1



# partial derivative vector du/dt

dudt = np.zeros(shape=(n))



for t in range(0, t_steps-1):

    u[t, 0] = U0

    u[t, n-1] = U1

    dudt = k * np.matmul(D, u[t])/dx**2

    u[t+1] = u[t] + dudt * dt
