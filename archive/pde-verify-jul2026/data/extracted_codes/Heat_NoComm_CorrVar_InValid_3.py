import numpy as np
from scipy.integrate import odeint
from scipy.linalg import toeplitz
foobar_10 = 1.0
foobar_11 = 100
foobar_2 = 1
foobar_9 = foobar_2 / foobar_11
foobar_15 = np.linspace(0, foobar_2, foobar_11)
foobar_3 = 0.0
foobar_4 = 0.1
foobar_13 = 100
foobar_12 = np.linspace(foobar_3, foobar_4, foobar_13)
foobar_16 = 1.0
foobar_17 = -1.0
foobar_6 = np.zeros(shape=(foobar_11,))
foobar_1 = toeplitz([-2.0, -1.0] + [0.0] * (foobar_11 - 2))
foobar_8 = np.zeros(shape=foobar_11)
foobar_8[0] = foobar_16
foobar_8[foobar_11 - 1] = foobar_17

def fn1(foobar_14, foobar_12, foobar_1, foobar_8, foobar_10, foobar_9):
    return foobar_10 / foobar_9 ** 2 * np.matmul(foobar_1, foobar_14) + foobar_10 / foobar_9 ** 2 * foobar_8
foobar_14 = odeint(fn1, foobar_6, foobar_12, args=(foobar_1, foobar_8, foobar_10, foobar_9))