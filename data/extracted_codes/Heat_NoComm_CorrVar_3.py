import numpy as np
from scipy.integrate import odeint
from scipy.linalg import toeplitz
foobar_11 = 0.005
foobar_12 = 100
foobar_4 = 1
foobar_10 = foobar_4 / foobar_12
foobar_16 = np.linspace(foobar_10, foobar_4 - foobar_10, foobar_12)
foobar_5 = 0.0
foobar_6 = 2.0
foobar_14 = 1000
foobar_9 = (foobar_6 - foobar_5) / foobar_14
foobar_13 = np.linspace(foobar_5, foobar_6, foobar_14)
foobar_2 = 3.0
foobar_3 = 0.0
foobar_7 = np.ones(shape=(foobar_12,)) * 0.0
foobar_1 = toeplitz([-2.0, 1.0] + [0.0] * (foobar_12 - 2))
foobar_1[foobar_12 - 1, foobar_12 - 1] = -1
foobar_1 = foobar_11 / foobar_10 ** 2 * foobar_1
foobar_8 = np.zeros(shape=foobar_12)
foobar_8[0] = foobar_11 / foobar_10 ** 2 * foobar_2
foobar_8[foobar_12 - 1] = 1 / foobar_10 * foobar_3

def fn1(foobar_15, foobar_13, foobar_1, foobar_8, foobar_11, foobar_10):
    return foobar_11 / foobar_10 ** 2 * np.matmul(foobar_1, foobar_15) + foobar_11 / foobar_10 ** 2 * foobar_8
foobar_15 = odeint(fn1, foobar_7, foobar_13, args=(foobar_1, foobar_8, foobar_11, foobar_10))