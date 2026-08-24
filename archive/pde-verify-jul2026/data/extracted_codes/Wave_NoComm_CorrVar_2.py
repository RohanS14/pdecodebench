import numpy as np

def fn1(foobar_25):
    return np.exp(-foobar_25 ** 2 / 0.01)

def fn2(foobar_25):
    if foobar_25 <= 0.7:
        return 1
    else:
        return 0.5
foobar_19 = 1
foobar_18 = 2
foobar_21 = 3
if foobar_18 not in [1, 2, 3]:
    foobar_19 = 0
    print('Please choose a correct left boundary condition')
if foobar_21 not in [1, 2, 3]:
    foobar_19 = 0
    print('Please choose a correct right boundary condition')
foobar_5 = 1.5
foobar_15 = 0.01
foobar_7 = int(foobar_5 / foobar_15)
foobar_10 = np.linspace(0, foobar_5, foobar_7 + 1)
foobar_4 = 4
foobar_14 = 0.01 * foobar_15
foobar_6 = int(foobar_4 / foobar_14)
foobar_8 = np.linspace(0, foobar_4, foobar_6 + 1)
foobar_11 = np.zeros(foobar_7 + 1, float)
for foobar_16 in range(0, foobar_7 + 1):
    foobar_11[foobar_16] = fn2(foobar_10[foobar_16])
foobar_12 = foobar_11[0]
foobar_13 = foobar_11[foobar_7]
foobar_1 = (foobar_14 / foobar_15) ** 2
foobar_2 = foobar_12 * (foobar_14 / foobar_15)
foobar_3 = foobar_13 * (foobar_14 / foobar_15)
if foobar_19:
    foobar_23 = np.zeros(foobar_7 + 1, float)
    foobar_22 = np.zeros(foobar_7 + 1, float)
    foobar_24 = np.zeros(foobar_7 + 1, float)
    foobar_20 = np.zeros(foobar_7 + 1, float)
    foobar_20[0:foobar_7 + 1] = foobar_11[0:foobar_7 + 1] ** 2
    foobar_9 = np.zeros((foobar_7 + 1, foobar_6 + 1), float)
    foobar_22[0:foobar_7 + 1] = fn1(foobar_10[0:foobar_7 + 1])
    foobar_9[:, 0] = foobar_22.copy()
    foobar_24[1:foobar_7] = foobar_22[1:foobar_7] + 0.5 * foobar_1 * (0.5 * (foobar_20[1:foobar_7] + foobar_20[2:foobar_7 + 1]) * (foobar_22[2:foobar_7 + 1] - foobar_22[1:foobar_7]) - 0.5 * (foobar_20[0:foobar_7 - 1] + foobar_20[1:foobar_7]) * (foobar_22[1:foobar_7] - foobar_22[0:foobar_7 - 1]))
    if foobar_18 == 1:
        foobar_24[0] = 0
    elif foobar_18 == 2:
        foobar_24[0] = foobar_22[0] + 0.5 * foobar_1 * (0.5 * (foobar_20[0] + foobar_20[0 + 1]) * (foobar_22[0 + 1] - foobar_22[0]) - 0.5 * (foobar_20[0] + foobar_20[0 + 1]) * (foobar_22[0] - foobar_22[0 + 1]))
    elif foobar_18 == 3:
        foobar_24[0] = foobar_22[1] + (foobar_2 - 1) / (foobar_2 + 1) * (foobar_24[1] - foobar_22[0])
    if foobar_21 == 1:
        foobar_24[foobar_7] = 0
    elif foobar_21 == 2:
        foobar_24[foobar_7] = foobar_22[foobar_7] + 0.5 * foobar_1 * (0.5 * (foobar_20[foobar_7 - 1] + foobar_20[foobar_7]) * (foobar_22[foobar_7 - 1] - foobar_22[foobar_7]) - 0.5 * (foobar_20[foobar_7 - 1] + foobar_20[foobar_7]) * (foobar_22[foobar_7] - foobar_22[foobar_16 - 1]))
    elif foobar_21 == 3:
        foobar_24[foobar_7] = foobar_22[foobar_7 - 1] + (foobar_3 - 1) / (foobar_3 + 1) * (foobar_24[foobar_7 - 1] - foobar_22[foobar_7])
    foobar_23 = foobar_22.copy()
    foobar_22 = foobar_24.copy()
    foobar_9[:, 1] = foobar_22.copy()
    for foobar_17 in range(1, foobar_6):
        foobar_24[1:foobar_7] = -foobar_23[1:foobar_7] + 2 * foobar_22[1:foobar_7] + foobar_1 * (0.5 * (foobar_20[1:foobar_7] + foobar_20[2:foobar_7 + 1]) * (foobar_22[2:foobar_7 + 1] - foobar_22[1:foobar_7]) - 0.5 * (foobar_20[0:foobar_7 - 1] + foobar_20[1:foobar_7]) * (foobar_22[1:foobar_7] - foobar_22[0:foobar_7 - 1]))
        if foobar_18 == 1:
            foobar_24[0] = 0
        elif foobar_18 == 2:
            foobar_24[0] = -foobar_23[0] + 2 * foobar_22[0] + foobar_1 * (0.5 * (foobar_20[0] + foobar_20[0 + 1]) * (foobar_22[0 + 1] - foobar_22[0]) - 0.5 * (foobar_20[0] + foobar_20[0 + 1]) * (foobar_22[0] - foobar_22[0 + 1]))
        elif foobar_18 == 3:
            foobar_24[0] = foobar_22[1] + (foobar_2 - 1) / (foobar_2 + 1) * (foobar_24[1] - foobar_22[0])
        if foobar_21 == 1:
            foobar_24[foobar_7] = 0
        elif foobar_21 == 2:
            foobar_24[foobar_7] = -foobar_23[foobar_7] + 2 * foobar_22[foobar_7] + foobar_1 * (0.5 * (foobar_20[foobar_7 - 1] + foobar_20[foobar_7]) * (foobar_22[foobar_7 - 1] - foobar_22[foobar_7]) - 0.5 * (foobar_20[foobar_7 - 1] + foobar_20[foobar_7]) * (foobar_22[foobar_7] - foobar_22[foobar_7 - 1]))
        elif foobar_21 == 3:
            foobar_24[foobar_7] = foobar_22[foobar_7 - 1] + (foobar_3 - 1) / (foobar_3 + 1) * (foobar_24[foobar_7 - 1] - foobar_22[foobar_7])
        foobar_23[:] = foobar_22.copy()
        foobar_22[:] = foobar_24.copy()
        foobar_9[:, foobar_17] = foobar_22.copy()