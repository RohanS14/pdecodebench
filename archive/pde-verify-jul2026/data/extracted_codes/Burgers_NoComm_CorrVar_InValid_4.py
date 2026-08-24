import numpy as np
from scipy.integrate import odeint
foobar_8 = 100
foobar_18 = 0
foobar_20 = 1
foobar_4 = (foobar_20 - foobar_18) / foobar_8
foobar_19 = np.linspace(foobar_18, foobar_20, foobar_8 + 1)
foobar_17 = foobar_19[0:foobar_8] + foobar_4 / 2
foobar_10 = 0
foobar_12 = 0.5
foobar_11 = 200
foobar_9 = np.linspace(foobar_10, foobar_12, foobar_11)
foobar_2 = foobar_9[1] - foobar_9[0]
foobar_16 = np.sin(2 * np.pi * foobar_17)

def fn1(foobar_14, foobar_9, foobar_4):
    foobar_8 = foobar_14.shape[0]
    foobar_5 = foobar_14 ** 2 / 2
    foobar_6 = 0.5 * (foobar_5[0:foobar_8 - 1] + foobar_5[1:foobar_8])
    foobar_6 = np.hstack([0, foobar_6, 0])
    foobar_3 = (foobar_6[0:foobar_8] - foobar_6[1:foobar_8 + 1]) / foobar_4
    return foobar_3
foobar_13 = odeint(fn1, foobar_16, foobar_9, args=(foobar_4,))
foobar_13 = np.zeros((foobar_11, foobar_8))
foobar_13[0] = foobar_16
foobar_15 = foobar_16.copy()
for foobar_7 in range(1, foobar_11):
    foobar_15 = foobar_15 + foobar_2 * fn1(foobar_15, foobar_9[foobar_7 - 1], foobar_4)
    foobar_13[foobar_7] = foobar_15