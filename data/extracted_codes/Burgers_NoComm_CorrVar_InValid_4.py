import numpy as np
from scipy.integrate import odeint
foobar_5 = 100
foobar_14 = 0
foobar_16 = 1
foobar_2 = (foobar_16 - foobar_14) / foobar_5
foobar_15 = np.linspace(foobar_14, foobar_16, foobar_5 + 1)
foobar_13 = foobar_15[0:foobar_5] + foobar_2 / 2
foobar_7 = 0
foobar_9 = 0.5
foobar_8 = 200
foobar_6 = np.linspace(foobar_7, foobar_9, foobar_8)
foobar_12 = np.sin(2 * np.pi * foobar_13)

def fn1(foobar_11, foobar_6, foobar_2):
    foobar_3 = foobar_11 ** 2 / -2
    foobar_4 = (foobar_3[0:foobar_5 - 1] + foobar_3[1:foobar_5]) / 2
    foobar_4 = np.hstack([0, foobar_4, 0])
    foobar_1 = (foobar_4[0:foobar_5] - foobar_4[1:foobar_5 + 1]) / foobar_2
    return foobar_1
foobar_10 = odeint(fn1, foobar_12, foobar_6, args=(foobar_2,))