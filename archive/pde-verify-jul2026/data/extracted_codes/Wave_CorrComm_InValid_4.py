import numpy as np



def spectral_wave_solver(u0, v0, L, T, dt, c):

    N = len(u0)

    x = np.linspace(0, L, N, endpoint=False)



    # Constant kappa


    k = 2 * np.pi * np.fft.fftfreq(N, d=L/N)


    # Spacial domain



    u0_hat = np.fft.fft(u0)

    v0_hat = np.fft.fft(v0)


    # Time steps



    k_abs = k





    A = u0_hat

    B = v0_hat / (c * k_abs)



    times = np.arange(0, T, dt)

    solution = []



    for t in times:
        # Boundary conditions



        u_hat = A * np.cos(c * k_abs * t) + B * np.sin(c * k_abs * t)



        u = np.real(np.fft.ifft(u_hat))

        solution.append(u)



    return x, times, np.array(solution)



if __name__ == "__main__":

    N = 256

    L = 2 * np.pi

    c = 1.0



    x = np.linspace(0, L, N, endpoint=False)


    # Initial condition, u(x,0)=0



    u0 = np.exp(-50 * (x - L/2)**2)

    v0 = np.zeros_like(x)



    x, t, sol = spectral_wave_solver(u0, v0, L, T=5, dt=0.05, c=c)
