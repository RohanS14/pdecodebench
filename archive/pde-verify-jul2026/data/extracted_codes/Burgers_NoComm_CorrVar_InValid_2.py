import numpy as np
from scipy.integrate import odeint
foobar_11 = 100
foobar_18 = np.zeros(shape=(foobar_11,))
foobar_29 = 0
foobar_31 = 1
foobar_4 = (foobar_31 - foobar_29) / foobar_11
foobar_30 = np.linspace(foobar_29, foobar_31, foobar_11 + 1)
foobar_28 = foobar_30[0:foobar_11] + foobar_4 / 2
foobar_15 = 0
foobar_17 = 0.5
foobar_16 = 200
foobar_14 = np.linspace(foobar_15, foobar_17, foobar_16)
foobar_2 = foobar_14[1] - foobar_14[0]
foobar_21 = np.sin(2 * np.pi * foobar_28)

def fn1(foobar_13):
    foobar_1 = foobar_13.copy()
    foobar_1[foobar_1 < 0] = 0
    foobar_12 = 2 * foobar_1 / (1 + foobar_1)
    foobar_12[~np.isfinite(foobar_1)] = 0
    return foobar_12

def fn2(foobar_26, foobar_27):
    foobar_6 = foobar_26 ** 2 / 2
    foobar_7 = foobar_27 ** 2 / 2
    foobar_9 = 0
    foobar_5 = np.zeros_like(foobar_26)
    foobar_5[foobar_26 > foobar_27] = np.maximum(foobar_6[foobar_26 > foobar_27], foobar_7[foobar_26 > foobar_27])
    foobar_5[foobar_26 <= foobar_27] = np.maximum(foobar_6[foobar_26 <= foobar_27], foobar_7[foobar_26 <= foobar_27])
    foobar_5[np.where((foobar_26 <= 0) & (foobar_27 > 0))] = foobar_9
    return foobar_5

def fn3(foobar_18, foobar_14, foobar_4):
    foobar_11 = len(foobar_18)
    foobar_20 = np.hstack([0, foobar_18[0:foobar_11 - 1]])
    foobar_22 = np.hstack([foobar_18[1:foobar_11], 0])
    foobar_23 = np.hstack([foobar_22[1:foobar_11], 0])
    foobar_13 = (foobar_22 - foobar_18) / (foobar_18 - foobar_20)
    foobar_12 = fn1(foobar_13)
    foobar_24 = foobar_18 + 0.5 * (foobar_18 - foobar_20) * foobar_12
    foobar_13 = (foobar_18 - foobar_22) / (foobar_22 - foobar_23)
    foobar_12 = fn1(foobar_13)
    foobar_25 = foobar_22 + 0.5 * (foobar_22 - foobar_23) * foobar_12
    foobar_8 = fn2(foobar_24, foobar_25)
    foobar_8 = np.hstack([0, foobar_8])
    foobar_3 = (foobar_8[0:foobar_11] - foobar_8[1:foobar_11 + 1]) / foobar_4
    return foobar_3
foobar_18 = np.zeros((foobar_16, foobar_11))
foobar_18[0] = foobar_21
foobar_19 = foobar_21.copy()
for foobar_10 in range(1, foobar_16):
    foobar_19 = foobar_19 + foobar_2 * fn3(foobar_19, foobar_14[foobar_10 - 1], foobar_4)
    foobar_18[foobar_10] = foobar_19