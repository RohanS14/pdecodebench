import numpy as np
from scipy.integrate import odeint
foobar_8 = 200
foobar_19 = 0
foobar_21 = 1
foobar_3 = (foobar_21 - foobar_19) / foobar_8
foobar_20 = np.linspace(foobar_19, foobar_21, foobar_8 + 1)
foobar_18 = foobar_20[0:foobar_8] + foobar_3 / 2
foobar_11 = 0
foobar_13 = 0.5
foobar_12 = 200
foobar_10 = np.linspace(foobar_11, foobar_13, foobar_12)
foobar_1 = foobar_10[1] - foobar_10[0]
foobar_17 = np.sin(2 * np.pi * foobar_18)

def fn1(foobar_15, foobar_10, foobar_3):
    foobar_8 = foobar_15.shape[0]
    foobar_4 = foobar_15 ** 2 / 2
    foobar_5 = np.zeros(shape=(foobar_8 + 1,))
    assert foobar_5.shape[0] == foobar_15.shape[0] + 1
    foobar_9 = np.zeros_like(foobar_5)
    foobar_9[1:foobar_8] = (foobar_15[0:foobar_8 - 1] + foobar_15[1:foobar_8]) / 2
    foobar_5[0] = 0
    for foobar_6 in range(1, foobar_8):
        if foobar_9[foobar_6] > 0:
            foobar_5[foobar_6] = foobar_4[foobar_6 - 1]
        else:
            foobar_5[foobar_6] = foobar_4[foobar_6]
    foobar_5[foobar_8] = 0
    foobar_2 = (foobar_5[0:foobar_8] - foobar_5[1:foobar_8 + 1]) / foobar_3
    return foobar_2
foobar_14 = np.zeros((foobar_12, foobar_8))
foobar_14[0] = foobar_17
foobar_16 = foobar_17.copy()
for foobar_7 in range(1, foobar_12):
    foobar_16 = foobar_16 + foobar_1 * fn1(foobar_16, foobar_10[foobar_7 - 1], foobar_3)
    foobar_14[foobar_7] = foobar_16