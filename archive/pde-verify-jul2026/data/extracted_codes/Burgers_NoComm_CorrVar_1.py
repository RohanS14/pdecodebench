import numpy as np
from scipy.integrate import odeint
from scipy.linalg import toeplitz
foobar_12 = 100
foobar_26 = 0
foobar_28 = 1
foobar_6 = (foobar_28 - foobar_26) / foobar_12
foobar_27 = np.linspace(foobar_26, foobar_28, foobar_12 + 1)
foobar_25 = foobar_27[0:foobar_12] + foobar_6 / 2
foobar_17 = 0
foobar_19 = 0.5
foobar_18 = 200
foobar_16 = np.linspace(foobar_17, foobar_19, foobar_18)
foobar_3 = foobar_16[1] - foobar_16[0]
foobar_22 = np.sin(2 * np.pi * foobar_25)

def fn1(foobar_1, foobar_2):
    return np.where(foobar_1 * foobar_2 > 0, np.sign(foobar_1) * np.minimum(np.abs(foobar_1), np.abs(foobar_2)), 0.0)

def fn2(foobar_20, foobar_16, foobar_6):
    foobar_12 = foobar_20.shape[0]
    foobar_5 = np.zeros_like(foobar_20)
    foobar_5[1:foobar_12 - 1] = (foobar_20[2:foobar_12] - foobar_20[0:foobar_12 - 2]) / (2 * foobar_6)
    foobar_14 = np.zeros_like(foobar_20)
    foobar_15 = np.zeros_like(foobar_20)
    foobar_14[1:foobar_12 - 1] = (foobar_20[1:foobar_12 - 1] - foobar_20[0:foobar_12 - 2]) / foobar_6
    foobar_15[1:foobar_12 - 1] = (foobar_20[2:foobar_12] - foobar_20[1:foobar_12 - 1]) / foobar_6
    foobar_5[1:foobar_12 - 1] = fn1(foobar_14[1:foobar_12 - 1], foobar_15[1:foobar_12 - 1])
    foobar_5[0] = (foobar_20[1] - foobar_20[0]) / foobar_6
    foobar_5[foobar_12 - 1] = (foobar_20[foobar_12 - 1] - foobar_20[foobar_12 - 2]) / foobar_6
    foobar_24 = foobar_20 + foobar_5 * foobar_6 / 2
    foobar_23 = foobar_20 - foobar_5 * foobar_6 / 2
    foobar_9 = foobar_24 ** 2 / 2
    foobar_8 = foobar_23 ** 2 / 2
    foobar_13 = np.zeros(shape=(foobar_12 + 1,))
    foobar_13[1:foobar_12] = (foobar_24[0:foobar_12 - 1] + foobar_23[1:foobar_12]) / 2
    foobar_7 = np.zeros(shape=(foobar_12 + 1,))
    assert foobar_7.shape[0] == foobar_20.shape[0] + 1
    foobar_7[0] = 0
    for foobar_10 in range(1, foobar_12):
        if foobar_13[foobar_10] > 0:
            foobar_7[foobar_10] = foobar_9[foobar_10 - 1]
        else:
            foobar_7[foobar_10] = foobar_8[foobar_10]
    foobar_7[foobar_12] = 0
    foobar_4 = (foobar_7[0:foobar_12] - foobar_7[1:foobar_12 + 1]) / foobar_6
    return foobar_4
foobar_20 = np.zeros((foobar_18, foobar_12))
foobar_20[0] = foobar_22
foobar_21 = foobar_22.copy()
for foobar_11 in range(1, foobar_18):
    foobar_21 = foobar_21 + foobar_3 * fn2(foobar_21, foobar_16[foobar_11 - 1], foobar_6)
    foobar_20[foobar_11] = foobar_21