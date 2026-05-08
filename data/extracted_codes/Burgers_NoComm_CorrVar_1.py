import numpy as np
from scipy.integrate import odeint
foobar_8 = 100
foobar_19 = 0
foobar_21 = 1
foobar_3 = (foobar_21 - foobar_19) / foobar_8
foobar_20 = np.linspace(foobar_19, foobar_21, foobar_8 + 1)
foobar_18 = foobar_20[0:foobar_8] + foobar_3 / 2
foobar_11 = 0
foobar_13 = 0.5
foobar_12 = 200
foobar_10 = np.linspace(foobar_11, foobar_13, foobar_12)
foobar_15 = np.sin(2 * np.pi * foobar_18)

def fn1(foobar_14, foobar_10, foobar_3):
    foobar_8 = foobar_14.shape[0]
    foobar_2 = np.zeros_like(foobar_14)
    foobar_2[1:foobar_8 - 1] = (foobar_14[2:foobar_8] - foobar_14[0:foobar_8 - 2]) / (2 * foobar_3)
    foobar_2[0] = (foobar_14[1] - foobar_14[0]) / foobar_3
    foobar_2[foobar_8 - 1] = (foobar_14[foobar_8 - 1] - foobar_14[foobar_8 - 2]) / foobar_3
    foobar_17 = foobar_14 + foobar_2 * foobar_3 / 2
    foobar_16 = foobar_14 - foobar_2 * foobar_3 / 2
    foobar_6 = foobar_17 ** 2 / 2
    foobar_5 = foobar_16 ** 2 / 2
    foobar_9 = np.zeros(shape=(foobar_8 + 1,))
    foobar_9[1:foobar_8] = (foobar_17[0:foobar_8 - 1] + foobar_16[1:foobar_8]) / 2
    foobar_4 = np.zeros(shape=(foobar_8 + 1,))
    assert foobar_4.shape[0] == foobar_14.shape[0] + 1
    foobar_4[0] = 0
    for foobar_7 in range(1, foobar_8):
        if foobar_9[foobar_7] > 0:
            foobar_4[foobar_7] = foobar_6[foobar_7 - 1]
        else:
            foobar_4[foobar_7] = foobar_5[foobar_7]
    foobar_4[foobar_8] = 0
    foobar_1 = (foobar_4[0:foobar_8] - foobar_4[1:foobar_8 + 1]) / foobar_3
    return foobar_1
foobar_14 = odeint(fn1, foobar_15, foobar_10, args=(foobar_3,))