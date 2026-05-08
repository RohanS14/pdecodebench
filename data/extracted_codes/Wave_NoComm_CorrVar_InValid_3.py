import numpy as np
from scipy.integrate import odeint
foobar_7 = 1
foobar_14 = 200
foobar_4 = 2 * np.pi
foobar_13 = foobar_4 / foobar_14
foobar_21 = np.linspace(0, foobar_4, foobar_14)
foobar_5 = 0
foobar_6 = 2 * np.pi
foobar_20 = 200
foobar_19 = np.linspace(foobar_5, foobar_6, foobar_20)
foobar_15 = np.sin(foobar_21 / 2) ** 16
foobar_18 = 0 * foobar_15
foobar_17 = np.hstack([foobar_15, foobar_18])
foobar_2 = np.zeros(foobar_14)
foobar_2 = foobar_2 + np.diag(1 / (2 * foobar_13) * np.ones(foobar_14 - 1), 1)
foobar_2 = foobar_2 + np.diag(-1 / (2 * foobar_13) * np.ones(foobar_14 - 1), -1)
foobar_2[0, foobar_14 - 1] = -1 / (2 * foobar_13)
foobar_2[foobar_14 - 1, 0] = 1 / (2 * foobar_13)
foobar_3 = np.zeros(foobar_14)
foobar_3 = foobar_3 + np.diag(1 / (2 * foobar_13) * np.ones(foobar_14 - 1), 1)
foobar_3 = foobar_3 + np.diag(-1 / (2 * foobar_13) * np.ones(foobar_14 - 1), -1)
foobar_3[0, foobar_14 - 1] = -1 / (2 * foobar_13)
foobar_3[foobar_14 - 1, 0] = 1 / (2 * foobar_13)
foobar_1 = np.zeros(shape=(2 * foobar_14, 2 * foobar_14))
foobar_1[:foobar_14, :foobar_14] = foobar_2
foobar_1[foobar_14:, foobar_14:] = foobar_3
foobar_1 = np.zeros(shape=(2 * foobar_14, 2 * foobar_14))
foobar_1[:foobar_14, :foobar_14] = foobar_2
foobar_1[foobar_14:, foobar_14:] = foobar_3

def fn1(foobar_16, foobar_19, foobar_1, foobar_14):
    foobar_8 = np.matmul(foobar_1, foobar_16)
    foobar_10 = foobar_8[:foobar_14]
    foobar_12 = foobar_8[foobar_14:]
    foobar_11 = foobar_10
    foobar_9 = foobar_12
    return np.hstack([np.zeros_like(foobar_9), np.zeros_like(foobar_11)])
foobar_16 = odeint(fn1, foobar_17, foobar_19, args=(foobar_1, foobar_14))