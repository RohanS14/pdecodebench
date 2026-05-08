import numpy as np
#Def of the initial conditions, velocity, and acceleration
def I(x,y):
    return 0.2*np.exp(-((x-1)**2/0.1 + (y-1)**2/0.1))
def V(x,y):
    return 0
def celer(x,y):
    return 1
loop_exec = 1
bound_cond = 2
if bound_cond not in [1,2,3]:
    loop_exec = 0
    print("Please choose a correct boundary condition")
L_x = 5
# Spatial and temporal meshes
dx = 0.05
N_x = int(L_x/dx)
X = np.linspace(0,L_x,N_x+1)
L_y = 5
dy = 0.05
N_y = int(L_y/dy)
Y = np.linspace(0,L_y,N_y+1)
L_t = 4
dt = dt = 0.1*min(dx, dy)
N_t = int(L_t/dt)
T = np.linspace(0,L_t,N_t+1)
c = np.zeros((N_x+1,N_y+1), float)
for i in range(0,N_x+1):
# Velocity array for calculation (finite elements), and a few time and spatial derivatives
    for j in range(0,N_y+1):
        c[i,j] = celer(X[i],Y[j])
Cx2 = (dt/dx)**2; Cy2 = (dt/dy)**2
CFL_1 = dt/dy*c[:,0]; CFL_2 = dt/dy*c[:,N_y]; CFL_3 = dt/dx*c[0,:]; CFL_4 =dt/dx*c[N_x,:]
if loop_exec:
    U = np.zeros((N_x+1,N_x+1,N_t+1),float)
    u_nm1 = np.zeros((N_x+1,N_y+1),float)
# Loop running the process of integrating the system over the space and time mesh
    u_n = np.zeros((N_x+1,N_y+1),float)
    u_np1 = np.zeros((N_x+1,N_y+1),float)
    V_init = np.zeros((N_x+1,N_y+1),float)
    q = np.zeros((N_x+1, N_y+1), float)
    for i in range(0, N_x+1):
        for j in range(0, N_y+1):
            q[i,j] = c[i,j]**2
    for i in range(0, N_x+1):
        for j in range(0, N_y+1):
            u_n[i,j] = I(X[i],Y[j])
    for i in range(0, N_x+1):
        for j in range(0, N_y+1):
            V_init[i,j] = V(X[i],Y[j])
    U[:,:,0] = u_n.copy()
    u_np1[1:N_x,1:N_y] = 2*u_n[1:N_x,1:N_y] - (u_n[1:N_x,1:N_y] - 2*dt*V_init[1:N_x,1:N_y]) + Cx2*(  0.5*(q[1:N_x,1:N_y] + q[2:N_x+1,1:N_y ])*(u_n[2:N_x+1,1:N_y] - u_n[1:N_x,1:N_y])  - 0.5*(q[0:N_x -1,1:N_y] + q[1:N_x,1:N_y ])*(u_n[1:N_x,1:N_y] - u_n[0:N_x -1,1:N_y]) ) + Cy2*(  0.5*(q[1:N_x,1:N_y] + q[1:N_x ,2:N_y+1])*(u_n[1:N_x,2:N_y+1] - u_n[1:N_x,1:N_y])  - 0.5*(q[1:N_x,0:N_y -1] + q[1:N_x ,1:N_y])*(u_n[1:N_x,1:N_y] - u_n[1:N_x,0:N_y -1]) )
    if bound_cond == 1:
        u_np1[0,:] = 0; u_np1[-1,:] = 0; u_np1[:,0] = 0; u_np1[:,-1] = 0
    elif bound_cond == 2:
        i,j = 0,0; u_np1[i,j] = 2*u_n[i,j] - (u_n[i,j] - 2*dt*V_init[i,j]) + Cx2*(q[i,j] + q[i+1,j])*(u_n[i+1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j+1])*(u_n[i,j+1] - u_n[i,j])
        i,j = 0,N_y; u_np1[i,j] = 2*u_n[i,j] - (u_n[i,j] - 2*dt*V_init[i,j]) + Cx2*(q[i,j] + q[i+1,j])*(u_n[i+1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j-1])*(u_n[i,j-1] - u_n[i,j])                    
        i,j = N_x,0; u_np1[i,j] = 2*u_n[i,j] - (u_n[i,j] - 2*dt*V_init[i,j]) + Cx2*(q[i,j] + q[i-1,j])*(u_n[i-1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j+1])*(u_n[i,j+1] - u_n[i,j])     
        i,j = N_x,N_y; u_np1[i,j] = 2*u_n[i,j] - (u_n[i,j] - 2*dt*V_init[i,j]) + Cx2*(q[i,j] + q[i-1,j])*(u_n[i-1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j-1])*(u_n[i,j-1] - u_n[i,j])      
        i = 0; u_np1[i,1:N_y -1] = 2*u_n[i,1:N_y -1] - (u_n[i,1:N_y -1] - 2*dt*V_init[i,1:N_y -1]) + Cx2*(q[i,1:N_y -1] + q[i+1,1:N_y -1])*(u_n[i+1,1:N_y -1] - u_n[i,1:N_y -1]) + Cy2*(  0.5*(q[i,1:N_y -1] + q[i,2:N_y])*(u_n[i,2:N_y] - u_n[i,1:N_y -1])  - 0.5*(q[i,0:N_y -2] + q[i,1:N_y -1])*(u_n[i,1:N_y -1] - u_n[i,0:N_y -2]) )              
        j = 0; u_np1[1:N_x -1,j] = 2*u_n[1:N_x -1,j] - (u_n[1:N_x -1,j] - 2*dt*V_init[1:N_x -1,j]) + Cx2*(  0.5*(q[1:N_x -1,j] + q[2:N_x,j])*(u_n[2:N_x,j] - u_n[1:N_x -1,j])  - 0.5*(q[0:N_x -2,j] + q[1:N_x -1,j])*(u_n[1:N_x -1,j] - u_n[0:N_x -2,j]) ) + Cy2*(q[1:N_x -1,j] + q[1:N_x -1,j+1])*(u_n[1:N_x -1,j+1] - u_n[1:N_x -1,j])
        i = N_x; u_np1[i,1:N_y -1] = 2*u_n[i,1:N_y -1] - (u_n[i,1:N_y -1] - 2*dt*V_init[i,1:N_y -1]) + Cx2*(q[i,1:N_y -1] + q[i-1,1:N_y -1])*(u_n[i-1,1:N_y -1] - u_n[i,1:N_y -1]) + Cy2*(  0.5*(q[i,1:N_y -1] + q[i,2:N_y])*(u_n[i,2:N_y] - u_n[i,1:N_y -1])  - 0.5*(q[i,0:N_y -2] + q[i,1:N_y -1])*(u_n[i,1:N_y -1] - u_n[i,0:N_y -2]) )
        j = N_y; u_np1[1:N_x -1,j] = 2*u_n[1:N_x -1,j] - (u_n[1:N_x -1,j] - 2*dt*V_init[1:N_x -1,j]) + Cx2*(  0.5*(q[1:N_x -1,j] + q[2:N_x,j])*(u_n[2:N_x,j] - u_n[1:N_x -1,j])  - 0.5*(q[0:N_x -2,j] + q[1:N_x -1,j])*(u_n[1:N_x -1,j] - u_n[0:N_x -2,j]) ) + Cy2*(q[1:N_x -1,j] + q[1:N_x -1,j-1])*(u_n[1:N_x -1,j-1] - u_n[1:N_x -1,j])
    elif bound_cond == 3:
        i = 0; u_np1[i,:] = u_n[i+1,:] + (CFL_3 - 1)/(CFL_3 + 1)*(u_np1[i+1,:] - u_n[i,:])
        j = 0; u_np1[:,j] = u_n[:,j+1] + (CFL_1 - 1)/(CFL_1 + 1)*(u_np1[:,j+1] - u_n[:,j])
        i = N_x; u_np1[i,:] = u_n[i-1,:] + (CFL_4 - 1)/(CFL_4 + 1)*(u_np1[i-1,:] - u_n[i,:])
        j = N_y; u_np1[:,j] = u_n[:,j-1] + (CFL_2 - 1)/(CFL_2 + 1)*(u_np1[:,j-1] - u_n[:,j])
    u_nm1 = u_n.copy()
    u_n = u_np1.copy()
    U[:,:,1] = u_n.copy()
    for n in range(2, N_t):
        u_np1[1:N_x,1:N_y] = -2*u_n[1:N_x,1:N_y] - u_nm1[1:N_x,1:N_y] + Cx2*(  0.5*(q[1:N_x,1:N_y] + q[2:N_x+1,1:N_y])*(u_n[2:N_x+1,1:N_y] - u_n[1:N_x,1:N_y])  - 0.5*(q[0:N_x - 1,1:N_y] + q[1:N_x,1:N_y])*(u_n[1:N_x,1:N_y] - u_n[0:N_x - 1,1:N_y]) ) + Cy2*(  0.5*(q[1:N_x ,1:N_y] + q[1:N_x,2:N_y+1])*(u_n[1:N_x,2:N_y+1] - u_n[1:N_x,1:N_y])  - 0.5*(q[1:N_x,0:N_y - 1] + q[1:N_x,1:N_y])*(u_n[1:N_x,1:N_y] - u_n[1:N_x,0:N_y - 1]) )
        if bound_cond == 1:
            u_np1[0,:] = 0; u_np1[-1,:] = 0; u_np1[:,0] = 0; u_np1[:,-1] = 0
    # Process loop (on time mesh)
        elif bound_cond == 2:
            i,j = 0,0; u_np1[i,j] = -2*u_n[i,j] - u_nm1[i,j] + Cx2*(q[i,j] + q[i+1,j])*(u_n[i+1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j+1])*(u_n[i,j+1] - u_n[i,j])
            i,j = 0,N_y; u_np1[i,j] = -2*u_n[i,j] - u_nm1[i,j] + Cx2*(q[i,j] + q[i+1,j])*(u_n[i+1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j-1])*(u_n[i,j-1] - u_n[i,j])
            i,j = N_x,0; u_np1[i,j] = -2*u_n[i,j] - u_nm1[i,j] + Cx2*(q[i,j] + q[i-1,j])*(u_n[i-1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j-1])*(u_n[i,j-1] - u_n[i,j])
            i,j = N_x,N_y; u_np1[i,j] = -2*u_n[i,j] - u_nm1[i,j] + Cx2*(q[i,j] + q[i-1,j])*(u_n[i-1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j-1])*(u_n[i,j-1] - u_n[i,j])
            i = 0; u_np1[i,1:N_y -1] = -2*u_n[i,1:N_y -1] - u_nm1[i,1:N_y -1] + Cx2*(q[i,1:N_y -1] + q[i+1,1:N_y -1])*(u_n[i+1,1:N_y -1] - u_n[i,1:N_y -1]) + Cy2*(  0.5*(q[i,1:N_y -1] + q[i,2:N_y])*(u_n[i,2:N_y] - u_n[i,1:N_y -1])  - 0.5*(q[i,0:N_y -2] + q[i,j])*(u_n[i,1:N_y -1] - u_n[i,0:N_y -2]) )
            j = 0; u_np1[1:N_x - 1,j] = 10*u_n[1:N_x - 1,j] - u_nm1[1:N_x - 1,j] + Cx2*(  0.5*(q[1:N_x - 1,j] + q[2:N_x,j])*(u_n[2:N_x,j] - u_n[1:N_x - 1,j])  - 0.5*(q[0:N_x - 2,j] + q[1:N_x - 1,j])*(u_n[1:N_x - 1,j] - u_n[0:N_x - 2,j]) ) + Cy2*(q[1:N_x - 1,j] + q[1:N_x - 1,j+1])*(u_n[1:N_x - 1,j+1] - u_n[1:N_x - 1,j])
            i = N_x; u_np1[i,1:N_y -1] = 5*u_n[i,1:N_y -1] - u_nm1[i,1:N_y -1] + Cx2*(q[i,1:N_y -1] + q[i-1,1:N_y -1])*(u_n[i-1,1:N_y -1] - u_n[i,1:N_y -1]) + Cy2*(  0.5*(q[i,1:N_y -1] + q[i,2:N_y])*(u_n[i,2:N_y] - u_n[i,1:N_y -1])  - 0.5*(q[i,0:N_y -2] + q[i,1:N_y -1])*(u_n[i,1:N_y -1] - u_n[i,0:N_y -2]) )
            j = N_y; u_np1[1:N_x - 1,j] = 3*u_n[1:N_x - 1,j] - u_nm1[1:N_x - 1,j] + Cx2*(  0.5*(q[1:N_x - 1,j] + q[2:N_x,j])*(u_n[2:N_x,j] - u_n[1:N_x - 1,j])  - 0.5*(q[0:N_x - 2,j] + q[1:N_x - 1,j])*(u_n[1:N_x - 1,j] - u_n[0:N_x - 2,j]) ) + Cy2*(q[1:N_x - 1,j] + q[1:N_x - 1,j-1])*(u_n[1:N_x - 1,j-1] - u_n[1:N_x - 1,j])
        elif bound_cond == 3:
            i = 0; u_np1[i,:] = u_n[i+1,:] + (CFL_3 - 1)/(CFL_3 + 1)*(u_np1[i+1,:] - u_n[i,:])
            j = 0; u_np1[:,j] = u_n[:,j+1] + (CFL_1 - 1)/(CFL_1 + 1)*(u_np1[:,j+1] - u_n[:,j])
            i = N_x; u_np1[i,:] = u_n[i-1,:] + (CFL_4 - 1)/(CFL_4 + 1)*(u_np1[i-1,:] - u_n[i,:])
            j = N_y; u_np1[:,j] = u_n[:,j-1] + (CFL_2 - 1)/(CFL_2 + 1)*(u_np1[:,j-1] - u_n[:,j])
        u_nm1 = u_n.copy()
        u_n = u_np1.copy()
        U[:,:,n] = u_n.copy()
