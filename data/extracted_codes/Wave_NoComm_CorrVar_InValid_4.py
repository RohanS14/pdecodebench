import numpy as np

def fn1(foobar_15, foobar_18, foobar_3, foobar_5, foobar_7, foobar_6):
    foobar_4 = len(foobar_15)
    foobar_20 = np.linspace(0, foobar_3, foobar_4, endpoint=False)
    foobar_8 = 2 * np.pi * np.fft.fftfreq(foobar_4, d=foobar_3 / foobar_4)
    foobar_16 = np.fft.fft(foobar_15)
    foobar_19 = np.fft.fft(foobar_18)
    foobar_9 = foobar_8
    foobar_1 = foobar_16
    foobar_2 = foobar_19 / (foobar_6 * foobar_9)
    foobar_13 = np.arange(0, foobar_5, foobar_7)
    foobar_11 = []
    for foobar_12 in foobar_13:
        foobar_17 = foobar_1 * np.cos(foobar_6 * foobar_9 * foobar_12) + foobar_2 * np.sin(foobar_6 * foobar_9 * foobar_12)
        foobar_14 = np.real(np.fft.ifft(foobar_17))
        foobar_11.append(foobar_14)
    return (foobar_20, foobar_13, np.array(foobar_11))
if __name__ == '__main__':
    foobar_4 = 256
    foobar_3 = 2 * np.pi
    foobar_6 = 1.0
    foobar_20 = np.linspace(0, foobar_3, foobar_4, endpoint=False)
    foobar_15 = np.exp(-50 * (foobar_20 - foobar_3 / 2) ** 2)
    foobar_18 = np.zeros_like(foobar_20)
    foobar_20, foobar_12, foobar_10 = fn1(foobar_15, foobar_18, foobar_3, 5, 0.05, foobar_6)