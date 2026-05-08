import numpy as np
from scipy.integrate import odeint
from scipy.linalg import toeplitz
foobar_10 = 1
foobar_11 = 100
foobar_2 = 1
foobar_9 = foobar_2 / foobar_11
foobar_15 = np.linspace(0, foobar_2, foobar_11)
foobar_3 = 0
foobar_4 = 0.1
foobar_13 = 100
foobar_12 = np.linspace(foobar_3, foobar_4, foobar_13)
foobar_5 = 1
foobar_7 = -1
foobar_6 = np.zeros(shape=(foobar_11,))
foobar_1 = toeplitz([-2.0, 1.0] + [0.0] * (foobar_11 - 2))
foobar_8 = np.zeros(shape=foobar_11)
foobar_8[0] = foobar_5
foobar_8[foobar_11 - 1] = foobar_7

def fn1(foobar_14, foobar_12, foobar_1, foobar_8, foobar_10, foobar_9):
    return foobar_10 / foobar_9 ** 2 * np.matmul(foobar_1, foobar_14) + foobar_10 / foobar_9 ** 2 * foobar_8
foobar_14 = odeint(fn1, foobar_6, foobar_12, args=(foobar_1, foobar_8, foobar_10, foobar_9))