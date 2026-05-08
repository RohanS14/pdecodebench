import numpy as np

def spectral_wave_solver(u0, v0, L, T, dt, c):
    N = len(u0)
    x = np.linspace(0, L, N, endpoint=False)

    # Wavenumbers
    k = 2 * np.pi * np.fft.fftfreq(N, d=L/N)

    # Fourier transform initial conditions
    u0_hat = np.fft.fft(u0)
    v0_hat = np.fft.fft(v0)

    # Avoid division by zero for k=0
    k_abs = np.abs(k)
    k_abs[0] = 1e-14

    A = u0_hat
    B = v0_hat / (c * k_abs)

    times = np.arange(0, T, dt)
    solution = []

    for t in times:
        # Exact solution in Fourier space
        u_hat = A * np.cos(c * k_abs * t) + B * np.sin(c * k_abs * t)

        u = np.real(np.fft.ifft(u_hat))
        solution.append(u)

    return x, times, np.array(solution)

if __name__ == "__main__":
    N = 256
    L = 2 * np.pi
    c = 1.0

    x = np.linspace(0, L, N, endpoint=False)

    # Initial conditions: Gaussian pulse
    u0 = np.exp(-50 * (x - L/2)**2)
    v0 = np.zeros_like(x)

    x, t, sol = spectral_wave_solver(u0, v0, L, T=5, dt=0.05, c=c)
