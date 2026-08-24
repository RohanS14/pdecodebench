import numpy as np

def fn1(foobar_31, foobar_32):
    return 0.2 * np.exp(-((foobar_31 - 1) ** 2 / 0.1 + (foobar_32 - 1) ** 2 / 0.1))

def fn2(foobar_31, foobar_32):
    return 0

def fn3(foobar_31, foobar_32):
    return 1
foobar_25 = 1
foobar_18 = 2
if foobar_18 not in [1, 2, 3]:
    foobar_25 = 0
    print('Please choose a correct boundary condition')
foobar_8 = 5
foobar_21 = 0.05
foobar_11 = int(foobar_8 / foobar_21)
foobar_16 = np.linspace(0, foobar_8, foobar_11 + 1)
foobar_9 = 5
foobar_22 = 0.05
foobar_12 = int(foobar_9 / foobar_22)
foobar_17 = np.linspace(0, foobar_9, foobar_12 + 1)
foobar_7 = 4
foobar_20 = foobar_20 = 0.1 * min(foobar_21, foobar_22)
foobar_10 = int(foobar_7 / foobar_20)
foobar_13 = np.linspace(0, foobar_7, foobar_10 + 1)
foobar_19 = np.zeros((foobar_11 + 1, foobar_12 + 1), float)
for foobar_23 in range(0, foobar_11 + 1):
    for foobar_24 in range(0, foobar_12 + 1):
        foobar_19[foobar_23, foobar_24] = fn3(foobar_16[foobar_23], foobar_17[foobar_24])
foobar_5 = (foobar_20 / foobar_21) ** 2
foobar_6 = (foobar_20 / foobar_22) ** 2
foobar_1 = foobar_20 / foobar_22 * foobar_19[:, 0]
foobar_2 = foobar_20 / foobar_22 * foobar_19[:, foobar_12]
foobar_3 = foobar_20 / foobar_21 * foobar_19[0, :]
foobar_4 = foobar_20 / foobar_21 * foobar_19[foobar_11, :]
if foobar_25:
    foobar_14 = np.zeros((foobar_11 + 1, foobar_11 + 1, foobar_10 + 1), float)
    foobar_29 = np.zeros((foobar_11 + 1, foobar_12 + 1), float)
    foobar_28 = np.zeros((foobar_11 + 1, foobar_12 + 1), float)
    foobar_30 = np.zeros((foobar_11 + 1, foobar_12 + 1), float)
    foobar_15 = np.zeros((foobar_11 + 1, foobar_12 + 1), float)
    foobar_27 = np.zeros((foobar_11 + 1, foobar_12 + 1), float)
    for foobar_23 in range(0, foobar_11 + 1):
        for foobar_24 in range(0, foobar_12 + 1):
            foobar_27[foobar_23, foobar_24] = foobar_19[foobar_23, foobar_24] ** 2
    for foobar_23 in range(0, foobar_11 + 1):
        for foobar_24 in range(0, foobar_12 + 1):
            foobar_28[foobar_23, foobar_24] = fn1(foobar_16[foobar_23], foobar_17[foobar_24])
    for foobar_23 in range(0, foobar_11 + 1):
        for foobar_24 in range(0, foobar_12 + 1):
            foobar_15[foobar_23, foobar_24] = fn2(foobar_16[foobar_23], foobar_17[foobar_24])
    foobar_14[:, :, 0] = foobar_28.copy()
    foobar_30[1:foobar_11, 1:foobar_12] = 2 * foobar_28[1:foobar_11, 1:foobar_12] - (foobar_28[1:foobar_11, 1:foobar_12] - 2 * foobar_20 * foobar_15[1:foobar_11, 1:foobar_12]) + foobar_5 * (0.5 * (foobar_27[1:foobar_11, 1:foobar_12] + foobar_27[2:foobar_11 + 1, 1:foobar_12]) * (foobar_28[2:foobar_11 + 1, 1:foobar_12] - foobar_28[1:foobar_11, 1:foobar_12]) - 0.5 * (foobar_27[0:foobar_11 - 1, 1:foobar_12] + foobar_27[1:foobar_11, 1:foobar_12]) * (foobar_28[1:foobar_11, 1:foobar_12] - foobar_28[0:foobar_11 - 1, 1:foobar_12])) + foobar_6 * (0.5 * (foobar_27[1:foobar_11, 1:foobar_12] + foobar_27[1:foobar_11, 2:foobar_12 + 1]) * (foobar_28[1:foobar_11, 2:foobar_12 + 1] - foobar_28[1:foobar_11, 1:foobar_12]) - 0.5 * (foobar_27[1:foobar_11, 0:foobar_12 - 1] + foobar_27[1:foobar_11, 1:foobar_12]) * (foobar_28[1:foobar_11, 1:foobar_12] - foobar_28[1:foobar_11, 0:foobar_12 - 1]))
    if foobar_18 == 1:
        foobar_30[0, :] = 0
        foobar_30[-1, :] = 0
        foobar_30[:, 0] = 0
        foobar_30[:, -1] = 0
    elif foobar_18 == 2:
        foobar_23, foobar_24 = (0, 0)
        foobar_30[foobar_23, foobar_24] = 2 * foobar_28[foobar_23, foobar_24] - (foobar_28[foobar_23, foobar_24] - 2 * foobar_20 * foobar_15[foobar_23, foobar_24]) + foobar_5 * (foobar_27[foobar_23, foobar_24] + foobar_27[foobar_23 + 1, foobar_24]) * (foobar_28[foobar_23 + 1, foobar_24] - foobar_28[foobar_23, foobar_24]) + foobar_6 * (foobar_27[foobar_23, foobar_24] + foobar_27[foobar_23, foobar_24 + 1]) * (foobar_28[foobar_23, foobar_24 + 1] - foobar_28[foobar_23, foobar_24])
        foobar_23, foobar_24 = (0, foobar_12)
        foobar_30[foobar_23, foobar_24] = 2 * foobar_28[foobar_23, foobar_24] - (foobar_28[foobar_23, foobar_24] - 2 * foobar_20 * foobar_15[foobar_23, foobar_24]) + foobar_5 * (foobar_27[foobar_23, foobar_24] + foobar_27[foobar_23 + 1, foobar_24]) * (foobar_28[foobar_23 + 1, foobar_24] - foobar_28[foobar_23, foobar_24]) + foobar_6 * (foobar_27[foobar_23, foobar_24] + foobar_27[foobar_23, foobar_24 - 1]) * (foobar_28[foobar_23, foobar_24 - 1] - foobar_28[foobar_23, foobar_24])
        foobar_23, foobar_24 = (foobar_11, 0)
        foobar_30[foobar_23, foobar_24] = 2 * foobar_28[foobar_23, foobar_24] - (foobar_28[foobar_23, foobar_24] - 2 * foobar_20 * foobar_15[foobar_23, foobar_24]) + foobar_5 * (foobar_27[foobar_23, foobar_24] + foobar_27[foobar_23 - 1, foobar_24]) * (foobar_28[foobar_23 - 1, foobar_24] - foobar_28[foobar_23, foobar_24]) + foobar_6 * (foobar_27[foobar_23, foobar_24] + foobar_27[foobar_23, foobar_24 + 1]) * (foobar_28[foobar_23, foobar_24 + 1] - foobar_28[foobar_23, foobar_24])
        foobar_23, foobar_24 = (foobar_11, foobar_12)
        foobar_30[foobar_23, foobar_24] = 2 * foobar_28[foobar_23, foobar_24] - (foobar_28[foobar_23, foobar_24] - 2 * foobar_20 * foobar_15[foobar_23, foobar_24]) + foobar_5 * (foobar_27[foobar_23, foobar_24] + foobar_27[foobar_23 - 1, foobar_24]) * (foobar_28[foobar_23 - 1, foobar_24] - foobar_28[foobar_23, foobar_24]) + foobar_6 * (foobar_27[foobar_23, foobar_24] + foobar_27[foobar_23, foobar_24 - 1]) * (foobar_28[foobar_23, foobar_24 - 1] - foobar_28[foobar_23, foobar_24])
        foobar_23 = 0
        foobar_30[foobar_23, 1:foobar_12 - 1] = 2 * foobar_28[foobar_23, 1:foobar_12 - 1] - (foobar_28[foobar_23, 1:foobar_12 - 1] - 2 * foobar_20 * foobar_15[foobar_23, 1:foobar_12 - 1]) + foobar_5 * (foobar_27[foobar_23, 1:foobar_12 - 1] + foobar_27[foobar_23 + 1, 1:foobar_12 - 1]) * (foobar_28[foobar_23 + 1, 1:foobar_12 - 1] - foobar_28[foobar_23, 1:foobar_12 - 1]) + foobar_6 * (0.5 * (foobar_27[foobar_23, 1:foobar_12 - 1] + foobar_27[foobar_23, 2:foobar_12]) * (foobar_28[foobar_23, 2:foobar_12] - foobar_28[foobar_23, 1:foobar_12 - 1]) - 0.5 * (foobar_27[foobar_23, 0:foobar_12 - 2] + foobar_27[foobar_23, 1:foobar_12 - 1]) * (foobar_28[foobar_23, 1:foobar_12 - 1] - foobar_28[foobar_23, 0:foobar_12 - 2]))
        foobar_24 = 0
        foobar_30[1:foobar_11 - 1, foobar_24] = 2 * foobar_28[1:foobar_11 - 1, foobar_24] - (foobar_28[1:foobar_11 - 1, foobar_24] - 2 * foobar_20 * foobar_15[1:foobar_11 - 1, foobar_24]) + foobar_5 * (0.5 * (foobar_27[1:foobar_11 - 1, foobar_24] + foobar_27[2:foobar_11, foobar_24]) * (foobar_28[2:foobar_11, foobar_24] - foobar_28[1:foobar_11 - 1, foobar_24]) - 0.5 * (foobar_27[0:foobar_11 - 2, foobar_24] + foobar_27[1:foobar_11 - 1, foobar_24]) * (foobar_28[1:foobar_11 - 1, foobar_24] - foobar_28[0:foobar_11 - 2, foobar_24])) + foobar_6 * (foobar_27[1:foobar_11 - 1, foobar_24] + foobar_27[1:foobar_11 - 1, foobar_24 + 1]) * (foobar_28[1:foobar_11 - 1, foobar_24 + 1] - foobar_28[1:foobar_11 - 1, foobar_24])
        foobar_23 = foobar_11
        foobar_30[foobar_23, 1:foobar_12 - 1] = 2 * foobar_28[foobar_23, 1:foobar_12 - 1] - (foobar_28[foobar_23, 1:foobar_12 - 1] - 2 * foobar_20 * foobar_15[foobar_23, 1:foobar_12 - 1]) + foobar_5 * (foobar_27[foobar_23, 1:foobar_12 - 1] + foobar_27[foobar_23 - 1, 1:foobar_12 - 1]) * (foobar_28[foobar_23 - 1, 1:foobar_12 - 1] - foobar_28[foobar_23, 1:foobar_12 - 1]) + foobar_6 * (0.5 * (foobar_27[foobar_23, 1:foobar_12 - 1] + foobar_27[foobar_23, 2:foobar_12]) * (foobar_28[foobar_23, 2:foobar_12] - foobar_28[foobar_23, 1:foobar_12 - 1]) - 0.5 * (foobar_27[foobar_23, 0:foobar_12 - 2] + foobar_27[foobar_23, 1:foobar_12 - 1]) * (foobar_28[foobar_23, 1:foobar_12 - 1] - foobar_28[foobar_23, 0:foobar_12 - 2]))
        foobar_24 = foobar_12
        foobar_30[1:foobar_11 - 1, foobar_24] = 2 * foobar_28[1:foobar_11 - 1, foobar_24] - (foobar_28[1:foobar_11 - 1, foobar_24] - 2 * foobar_20 * foobar_15[1:foobar_11 - 1, foobar_24]) + foobar_5 * (0.5 * (foobar_27[1:foobar_11 - 1, foobar_24] + foobar_27[2:foobar_11, foobar_24]) * (foobar_28[2:foobar_11, foobar_24] - foobar_28[1:foobar_11 - 1, foobar_24]) - 0.5 * (foobar_27[0:foobar_11 - 2, foobar_24] + foobar_27[1:foobar_11 - 1, foobar_24]) * (foobar_28[1:foobar_11 - 1, foobar_24] - foobar_28[0:foobar_11 - 2, foobar_24])) + foobar_6 * (foobar_27[1:foobar_11 - 1, foobar_24] + foobar_27[1:foobar_11 - 1, foobar_24 - 1]) * (foobar_28[1:foobar_11 - 1, foobar_24 - 1] - foobar_28[1:foobar_11 - 1, foobar_24])
    elif foobar_18 == 3:
        foobar_23 = 0
        foobar_30[foobar_23, :] = foobar_28[foobar_23 + 1, :] + (foobar_3 - 1) / (foobar_3 + 1) * (foobar_30[foobar_23 + 1, :] - foobar_28[foobar_23, :])
        foobar_24 = 0
        foobar_30[:, foobar_24] = foobar_28[:, foobar_24 + 1] + (foobar_1 - 1) / (foobar_1 + 1) * (foobar_30[:, foobar_24 + 1] - foobar_28[:, foobar_24])
        foobar_23 = foobar_11
        foobar_30[foobar_23, :] = foobar_28[foobar_23 - 1, :] + (foobar_4 - 1) / (foobar_4 + 1) * (foobar_30[foobar_23 - 1, :] - foobar_28[foobar_23, :])
        foobar_24 = foobar_12
        foobar_30[:, foobar_24] = foobar_28[:, foobar_24 - 1] + (foobar_2 - 1) / (foobar_2 + 1) * (foobar_30[:, foobar_24 - 1] - foobar_28[:, foobar_24])
    foobar_29 = foobar_28.copy()
    foobar_28 = foobar_30.copy()
    foobar_14[:, :, 1] = foobar_28.copy()
    for foobar_26 in range(2, foobar_10):
        foobar_30[1:foobar_11, 1:foobar_12] = -2 * foobar_28[1:foobar_11, 1:foobar_12] - foobar_29[1:foobar_11, 1:foobar_12] + foobar_5 * (0.5 * (foobar_27[1:foobar_11, 1:foobar_12] + foobar_27[2:foobar_11 + 1, 1:foobar_12]) * (foobar_28[2:foobar_11 + 1, 1:foobar_12] - foobar_28[1:foobar_11, 1:foobar_12]) - 0.5 * (foobar_27[0:foobar_11 - 1, 1:foobar_12] + foobar_27[1:foobar_11, 1:foobar_12]) * (foobar_28[1:foobar_11, 1:foobar_12] - foobar_28[0:foobar_11 - 1, 1:foobar_12])) + foobar_6 * (0.5 * (foobar_27[1:foobar_11, 1:foobar_12] + foobar_27[1:foobar_11, 2:foobar_12 + 1]) * (foobar_28[1:foobar_11, 2:foobar_12 + 1] - foobar_28[1:foobar_11, 1:foobar_12]) - 0.5 * (foobar_27[1:foobar_11, 0:foobar_12 - 1] + foobar_27[1:foobar_11, 1:foobar_12]) * (foobar_28[1:foobar_11, 1:foobar_12] - foobar_28[1:foobar_11, 0:foobar_12 - 1]))
        if foobar_18 == 1:
            foobar_30[0, :] = 0
            foobar_30[-1, :] = 0
            foobar_30[:, 0] = 0
            foobar_30[:, -1] = 0
        elif foobar_18 == 2:
            foobar_23, foobar_24 = (0, 0)
            foobar_30[foobar_23, foobar_24] = -2 * foobar_28[foobar_23, foobar_24] - foobar_29[foobar_23, foobar_24] + foobar_5 * (foobar_27[foobar_23, foobar_24] + foobar_27[foobar_23 + 1, foobar_24]) * (foobar_28[foobar_23 + 1, foobar_24] - foobar_28[foobar_23, foobar_24]) + foobar_6 * (foobar_27[foobar_23, foobar_24] + foobar_27[foobar_23, foobar_24 + 1]) * (foobar_28[foobar_23, foobar_24 + 1] - foobar_28[foobar_23, foobar_24])
            foobar_23, foobar_24 = (0, foobar_12)
            foobar_30[foobar_23, foobar_24] = -2 * foobar_28[foobar_23, foobar_24] - foobar_29[foobar_23, foobar_24] + foobar_5 * (foobar_27[foobar_23, foobar_24] + foobar_27[foobar_23 + 1, foobar_24]) * (foobar_28[foobar_23 + 1, foobar_24] - foobar_28[foobar_23, foobar_24]) + foobar_6 * (foobar_27[foobar_23, foobar_24] + foobar_27[foobar_23, foobar_24 - 1]) * (foobar_28[foobar_23, foobar_24 - 1] - foobar_28[foobar_23, foobar_24])
            foobar_23, foobar_24 = (foobar_11, 0)
            foobar_30[foobar_23, foobar_24] = -2 * foobar_28[foobar_23, foobar_24] - foobar_29[foobar_23, foobar_24] + foobar_5 * (foobar_27[foobar_23, foobar_24] + foobar_27[foobar_23 - 1, foobar_24]) * (foobar_28[foobar_23 - 1, foobar_24] - foobar_28[foobar_23, foobar_24]) + foobar_6 * (foobar_27[foobar_23, foobar_24] + foobar_27[foobar_23, foobar_24 - 1]) * (foobar_28[foobar_23, foobar_24 - 1] - foobar_28[foobar_23, foobar_24])
            foobar_23, foobar_24 = (foobar_11, foobar_12)
            foobar_30[foobar_23, foobar_24] = -2 * foobar_28[foobar_23, foobar_24] - foobar_29[foobar_23, foobar_24] + foobar_5 * (foobar_27[foobar_23, foobar_24] + foobar_27[foobar_23 - 1, foobar_24]) * (foobar_28[foobar_23 - 1, foobar_24] - foobar_28[foobar_23, foobar_24]) + foobar_6 * (foobar_27[foobar_23, foobar_24] + foobar_27[foobar_23, foobar_24 - 1]) * (foobar_28[foobar_23, foobar_24 - 1] - foobar_28[foobar_23, foobar_24])
            foobar_23 = 0
            foobar_30[foobar_23, 1:foobar_12 - 1] = -2 * foobar_28[foobar_23, 1:foobar_12 - 1] - foobar_29[foobar_23, 1:foobar_12 - 1] + foobar_5 * (foobar_27[foobar_23, 1:foobar_12 - 1] + foobar_27[foobar_23 + 1, 1:foobar_12 - 1]) * (foobar_28[foobar_23 + 1, 1:foobar_12 - 1] - foobar_28[foobar_23, 1:foobar_12 - 1]) + foobar_6 * (0.5 * (foobar_27[foobar_23, 1:foobar_12 - 1] + foobar_27[foobar_23, 2:foobar_12]) * (foobar_28[foobar_23, 2:foobar_12] - foobar_28[foobar_23, 1:foobar_12 - 1]) - 0.5 * (foobar_27[foobar_23, 0:foobar_12 - 2] + foobar_27[foobar_23, foobar_24]) * (foobar_28[foobar_23, 1:foobar_12 - 1] - foobar_28[foobar_23, 0:foobar_12 - 2]))
            foobar_24 = 0
            foobar_30[1:foobar_11 - 1, foobar_24] = 10 * foobar_28[1:foobar_11 - 1, foobar_24] - foobar_29[1:foobar_11 - 1, foobar_24] + foobar_5 * (0.5 * (foobar_27[1:foobar_11 - 1, foobar_24] + foobar_27[2:foobar_11, foobar_24]) * (foobar_28[2:foobar_11, foobar_24] - foobar_28[1:foobar_11 - 1, foobar_24]) - 0.5 * (foobar_27[0:foobar_11 - 2, foobar_24] + foobar_27[1:foobar_11 - 1, foobar_24]) * (foobar_28[1:foobar_11 - 1, foobar_24] - foobar_28[0:foobar_11 - 2, foobar_24])) + foobar_6 * (foobar_27[1:foobar_11 - 1, foobar_24] + foobar_27[1:foobar_11 - 1, foobar_24 + 1]) * (foobar_28[1:foobar_11 - 1, foobar_24 + 1] - foobar_28[1:foobar_11 - 1, foobar_24])
            foobar_23 = foobar_11
            foobar_30[foobar_23, 1:foobar_12 - 1] = 5 * foobar_28[foobar_23, 1:foobar_12 - 1] - foobar_29[foobar_23, 1:foobar_12 - 1] + foobar_5 * (foobar_27[foobar_23, 1:foobar_12 - 1] + foobar_27[foobar_23 - 1, 1:foobar_12 - 1]) * (foobar_28[foobar_23 - 1, 1:foobar_12 - 1] - foobar_28[foobar_23, 1:foobar_12 - 1]) + foobar_6 * (0.5 * (foobar_27[foobar_23, 1:foobar_12 - 1] + foobar_27[foobar_23, 2:foobar_12]) * (foobar_28[foobar_23, 2:foobar_12] - foobar_28[foobar_23, 1:foobar_12 - 1]) - 0.5 * (foobar_27[foobar_23, 0:foobar_12 - 2] + foobar_27[foobar_23, 1:foobar_12 - 1]) * (foobar_28[foobar_23, 1:foobar_12 - 1] - foobar_28[foobar_23, 0:foobar_12 - 2]))
            foobar_24 = foobar_12
            foobar_30[1:foobar_11 - 1, foobar_24] = 3 * foobar_28[1:foobar_11 - 1, foobar_24] - foobar_29[1:foobar_11 - 1, foobar_24] + foobar_5 * (0.5 * (foobar_27[1:foobar_11 - 1, foobar_24] + foobar_27[2:foobar_11, foobar_24]) * (foobar_28[2:foobar_11, foobar_24] - foobar_28[1:foobar_11 - 1, foobar_24]) - 0.5 * (foobar_27[0:foobar_11 - 2, foobar_24] + foobar_27[1:foobar_11 - 1, foobar_24]) * (foobar_28[1:foobar_11 - 1, foobar_24] - foobar_28[0:foobar_11 - 2, foobar_24])) + foobar_6 * (foobar_27[1:foobar_11 - 1, foobar_24] + foobar_27[1:foobar_11 - 1, foobar_24 - 1]) * (foobar_28[1:foobar_11 - 1, foobar_24 - 1] - foobar_28[1:foobar_11 - 1, foobar_24])
        elif foobar_18 == 3:
            foobar_23 = 0
            foobar_30[foobar_23, :] = foobar_28[foobar_23 + 1, :] + (foobar_3 - 1) / (foobar_3 + 1) * (foobar_30[foobar_23 + 1, :] - foobar_28[foobar_23, :])
            foobar_24 = 0
            foobar_30[:, foobar_24] = foobar_28[:, foobar_24 + 1] + (foobar_1 - 1) / (foobar_1 + 1) * (foobar_30[:, foobar_24 + 1] - foobar_28[:, foobar_24])
            foobar_23 = foobar_11
            foobar_30[foobar_23, :] = foobar_28[foobar_23 - 1, :] + (foobar_4 - 1) / (foobar_4 + 1) * (foobar_30[foobar_23 - 1, :] - foobar_28[foobar_23, :])
            foobar_24 = foobar_12
            foobar_30[:, foobar_24] = foobar_28[:, foobar_24 - 1] + (foobar_2 - 1) / (foobar_2 + 1) * (foobar_30[:, foobar_24 - 1] - foobar_28[:, foobar_24])
        foobar_29 = foobar_28.copy()
        foobar_28 = foobar_30.copy()
        foobar_14[:, :, foobar_26] = foobar_28.copy()