import numpy as np
from scipy.integrate import odeint
foobar_9 = 100
foobar_16 = np.zeros(shape=(foobar_9,))
foobar_26 = 0
foobar_28 = 1
foobar_3 = (foobar_28 - foobar_26) / foobar_9
foobar_27 = np.linspace(foobar_26, foobar_28, foobar_9 + 1)
foobar_25 = foobar_27[0:foobar_9] + foobar_3 / 2
foobar_13 = 0
foobar_15 = 0.5
foobar_14 = 200
foobar_12 = np.linspace(foobar_13, foobar_15, foobar_14)
foobar_18 = np.sin(2 * np.pi * foobar_25)

def fn1(foobar_11):
    foobar_1 = foobar_11.copy()
    foobar_1[foobar_1 < 0] = 0
    foobar_10 = 2 * foobar_1 / (1 + foobar_1)
    foobar_10[~np.isfinite(foobar_1)] = 0
    return foobar_10

def fn2(foobar_23, foobar_24):
    foobar_5 = foobar_23 ** 2 / 2
    foobar_6 = foobar_24 ** 2 / 2
    foobar_8 = 0
    foobar_4 = np.zeros_like(foobar_23)
    foobar_4[foobar_23 > foobar_24] = np.maximum(foobar_5[foobar_23 > foobar_24], foobar_6[foobar_23 > foobar_24])
    foobar_4[foobar_23 <= foobar_24] = np.minimum(foobar_5[foobar_23 <= foobar_24], foobar_6[foobar_23 <= foobar_24])
    foobar_4[np.where((foobar_23 <= 0) & (foobar_24 > 0))] = foobar_8
    return foobar_4

def fn3(foobar_16, foobar_12, foobar_3):
    foobar_9 = len(foobar_16)
    foobar_17 = np.hstack([0, foobar_16[0:foobar_9 - 1]])
    foobar_19 = np.hstack([foobar_16[1:foobar_9], 0])
    foobar_20 = np.hstack([foobar_19[1:foobar_9], 0])
    foobar_11 = (foobar_19 - foobar_16) / (foobar_16 - foobar_17)
    foobar_10 = fn1(foobar_11)
    foobar_21 = foobar_16 + 0.5 * (foobar_16 - foobar_17) * foobar_10
    foobar_11 = (foobar_16 - foobar_19) / (foobar_19 - foobar_20)
    foobar_10 = fn1(foobar_11)
    foobar_22 = foobar_19 + 0.5 * (foobar_19 - foobar_20) * foobar_10
    foobar_7 = fn2(foobar_21, foobar_22)
    foobar_7 = np.hstack([0, foobar_7])
    foobar_2 = (foobar_7[0:foobar_9] - foobar_7[1:foobar_9 + 1]) / foobar_3
    return foobar_2
foobar_16 = odeint(fn3, foobar_18, -foobar_12, args=(foobar_3,))