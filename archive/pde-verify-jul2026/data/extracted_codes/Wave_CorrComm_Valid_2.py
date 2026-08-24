import numpy as np

# Wavenumber grids

def I(x):

    return np.exp(-(x)**2/0.01)

def celer(x):

    if x <=0.7:

        return 1

    else:

        return 0.5

loop_exec = 1

left_bound_cond = 2

right_bound_cond = 3

if left_bound_cond not in [1,2,3]:

    loop_exec = 0

    print("Please choose a correct left boundary condition")

if right_bound_cond not in [1,2,3]:

    loop_exec = 0

    print("Please choose a correct right boundary condition")

# Anti-aliasing filter (2/3 rule)

L_x = 1.5

dx = 0.01

N_x = int(L_x/dx)

X = np.linspace(0,L_x,N_x+1)

L_t = 4

dt = 0.01*dx

N_t = int(L_t/dt)

T = np.linspace(0,L_t,N_t+1)

# Vorticity -> velocity

c = np.zeros(N_x+1, float)

for i in range(0,N_x+1):

    c[i] = celer(X[i])

c_1 = c[0]; c_2 = c[N_x]; C2 = (dt/dx)**2; CFL_1 = c_1*(dt/dx); CFL_2 = c_2*(dt/dx)

# Initial Conditions

if loop_exec:

    u_jm1 = np.zeros(N_x+1,float)

    u_j = np.zeros(N_x+1,float)

    u_jp1 = np.zeros(N_x+1,float)

    q = np.zeros(N_x+1,float)

    q[0:N_x+1] = c[0:N_x+1]**2

    U = np.zeros((N_x+1,N_t+1),float)

    u_j[0:N_x+1] = I(X[0:N_x+1])

    U[:,0] = u_j.copy()

    u_jp1[1:N_x] = u_j[1:N_x] + 0.5*C2*( 0.5*(q[1:N_x] + q[2:N_x+1])*(u_j[2:N_x+1] - u_j[1:N_x]) - 0.5*(q[0:N_x-1] + q[1:N_x])*(u_j[1:N_x] - u_j[0:N_x-1]))

    if left_bound_cond == 1:

        u_jp1[0] = 0

    elif left_bound_cond == 2:

        u_jp1[0] = u_j[0] + 0.5*C2*( 0.5*(q[0] + q[0+1])*(u_j[0+1] - u_j[0]) - 0.5*(q[0] + q[0+1])*(u_j[0] - u_j[0+1]))

    elif left_bound_cond == 3:

        u_jp1[0] = u_j[1] + (CFL_1 -1)/(CFL_1 + 1)*( u_jp1[1] - u_j[0])

    if right_bound_cond == 1:

        u_jp1[N_x] = 0

    elif right_bound_cond == 2:

        u_jp1[N_x] =  u_j[N_x] + 0.5*C2*( 0.5*(q[N_x-1] + q[N_x])*(u_j[N_x-1] - u_j[N_x]) - 0.5*(q[N_x-1] + q[N_x])*(u_j[N_x] - u_j[i-1]))

    elif right_bound_cond == 3:

        u_jp1[N_x] = u_j[N_x-1] + (CFL_2 -1)/(CFL_2 + 1)*(u_jp1[N_x-1] - u_j[N_x])

    u_jm1 = u_j.copy()

    u_j = u_jp1.copy()

    U[:,1] = u_j.copy()

    # Explict terms (advection only)

    for j in range(1, N_t):

        u_jp1[1:N_x] = -u_jm1[1:N_x] + 2*u_j[1:N_x] + C2*( 0.5*(q[1:N_x] + q[2:N_x+1])*(u_j[2:N_x+1] - u_j[1:N_x]) - 0.5*(q[0:N_x-1] + q[1:N_x])*(u_j[1:N_x] - u_j[0:N_x-1]))

        if left_bound_cond == 1:

            u_jp1[0] = 0

        elif left_bound_cond == 2:

            u_jp1[0] = -u_jm1[0] + 2*u_j[0] + C2*( 0.5*(q[0] + q[0+1])*(u_j[0+1] - u_j[0]) - 0.5*(q[0] + q[0+1])*(u_j[0] - u_j[0+1]))       

        elif left_bound_cond == 3:

            u_jp1[0] = u_j[1] + (CFL_1 -1)/(CFL_1 + 1)*( u_jp1[1] - u_j[0])

        if right_bound_cond == 1:

            u_jp1[N_x] = 0

        elif right_bound_cond == 2:

            u_jp1[N_x] = -u_jm1[N_x] + 2*u_j[N_x] + C2*( 0.5*(q[N_x-1] + q[N_x])*(u_j[N_x-1] - u_j[N_x]) - 0.5*(q[N_x-1] + q[N_x])*(u_j[N_x] - u_j[N_x-1]))

        elif right_bound_cond == 3:

            u_jp1[N_x] = u_j[N_x-1] + (CFL_2 -1)/(CFL_2 + 1)*(u_jp1[N_x-1] - u_j[N_x])

        u_jm1[:] = u_j.copy()

        u_j[:] = u_jp1.copy()

        U[:,j] = u_j.copy()
