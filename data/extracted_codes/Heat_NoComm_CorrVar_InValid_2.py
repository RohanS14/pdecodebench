import numpy as np
from scipy.integrate import odeint
from scipy.linalg import toeplitz
foobar_6 = 0.0012
foobar_8 = -0.0015
foobar_4 = 300
foobar_18 = 0.0
foobar_19 = 1.0
foobar_11 = 100
foobar_10 = (foobar_19 - foobar_18) / foobar_11
foobar_17 = np.linspace(foobar_18, foobar_19, foobar_11)
foobar_13 = 0.0
foobar_15 = 300.0
foobar_14 = 300
foobar_9 = (foobar_15 - foobar_13) / foobar_14
foobar_12 = np.linspace(foobar_13, foobar_15, foobar_14)
foobar_2 = 325
foobar_3 = 10
foobar_5 = np.ones(shape=(foobar_11,)) * 325
foobar_1 = toeplitz([-2 * foobar_6 / foobar_10 ** 2 + foobar_8, foobar_6 / foobar_10 ** 2] + [0.0] * (foobar_11 - 2))
foobar_1[foobar_11 - 1, foobar_11 - 1] = -foobar_6 / foobar_10 ** 2 + foobar_8
foobar_7 = np.ones(shape=foobar_11) * -foobar_8 * foobar_4
foobar_7[0] = -foobar_8 * foobar_4 + foobar_6 / foobar_10 ** 2 * foobar_2
foobar_7[foobar_11 - 1] = -foobar_8 * foobar_4 + foobar_3 / foobar_10

def fn1(foobar_16, foobar_12, foobar_1, foobar_7):
    return np.matmul(-foobar_1, foobar_16 + np.ones_like(foobar_16)) + foobar_7
foobar_16 = odeint(fn1, foobar_5, foobar_12, args=(foobar_1, foobar_7))