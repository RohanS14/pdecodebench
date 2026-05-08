import numpy as np
from scipy.integrate import odeint
foobar_6 = 200
foobar_16 = 0
foobar_18 = 1
foobar_2 = (foobar_18 - foobar_16) / foobar_6
foobar_17 = np.linspace(foobar_16, foobar_18, foobar_6 + 1)
foobar_15 = foobar_17[0:foobar_6] + foobar_2 / 2
foobar_9 = 0
foobar_11 = 0.5
foobar_10 = 200
foobar_8 = np.linspace(foobar_9, foobar_11, foobar_10)
foobar_14 = np.sin(2 * np.pi * foobar_15)

def fn1(foobar_13, foobar_8, foobar_2):
    foobar_6 = foobar_13.shape[0]
    foobar_3 = foobar_13 ** 2 / -2
    foobar_4 = np.zeros(shape=(foobar_6 + 1,))
    foobar_7 = np.zeros_like(foobar_4)
    foobar_7[1:foobar_6] = (foobar_13[0:foobar_6 - 1] + foobar_13[1:foobar_6]) / 2
    foobar_4[0] = 0
    for foobar_5 in range(1, foobar_6):
        if foobar_7[foobar_5] > 0:
            foobar_4[foobar_5] = foobar_3[foobar_5 - 1]
        else:
            foobar_4[foobar_5] = foobar_3[foobar_5]
    foobar_4[foobar_6] = 0
    foobar_1 = (foobar_4[0:foobar_6] - foobar_4[1:foobar_6 + 1]) / foobar_2
    return foobar_1
foobar_12 = odeint(fn1, foobar_14, foobar_8, args=(foobar_2,))