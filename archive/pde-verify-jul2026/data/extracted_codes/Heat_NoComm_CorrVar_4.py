import numpy as np
foobar_13 = 0.001
foobar_14 = 50
foobar_3 = 1
foobar_11 = foobar_3 / foobar_14
foobar_18 = np.linspace(0, foobar_3, foobar_14)
foobar_4 = 0
foobar_5 = 240
foobar_16 = (foobar_5 - foobar_4) * 10
foobar_9 = (foobar_5 - foobar_4) / foobar_16
foobar_6 = 100
foobar_7 = 0
foobar_8 = 0
foobar_1 = foobar_13 * foobar_9 / foobar_11 ** 2
print(f'CFL: {foobar_1:0.3f} < 0.5')
assert foobar_1 < 0.5
foobar_17 = np.ones(shape=(foobar_16, foobar_14)) * foobar_8
foobar_2 = np.zeros(shape=(foobar_14, foobar_14))
for foobar_12 in range(1, foobar_14 - 1):
    foobar_2[foobar_12, foobar_12 - 1] = 1
    foobar_2[foobar_12, foobar_12] = -2
    foobar_2[foobar_12, foobar_12 + 1] = 1
foobar_10 = np.zeros(shape=foobar_14)
for foobar_15 in range(0, foobar_16 - 1):
    foobar_17[foobar_15, 0] = foobar_6
    foobar_17[foobar_15, foobar_14 - 1] = foobar_7
    foobar_10 = foobar_13 * np.matmul(foobar_2, foobar_17[foobar_15]) / foobar_11 ** 2
    foobar_17[foobar_15 + 1] = foobar_17[foobar_15] + foobar_10 * foobar_9