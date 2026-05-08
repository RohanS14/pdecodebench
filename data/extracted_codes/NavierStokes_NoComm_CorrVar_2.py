import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

def fn1(foobar_17, foobar_6, foobar_7, foobar_8, foobar_19, foobar_22):
    foobar_4 = numpy.zeros_like(foobar_19)
    foobar_4[1:-1, 1:-1] = foobar_17 * (1 / foobar_6 * ((foobar_19[1:-1, 2:] - foobar_19[1:-1, 0:-2]) / (2 * foobar_7) + (foobar_22[2:, 1:-1] - foobar_22[0:-2, 1:-1]) / (2 * foobar_8)) - ((foobar_19[1:-1, 2:] - foobar_19[1:-1, 0:-2]) / (2 * foobar_7)) ** 2 - 2 * ((foobar_19[2:, 1:-1] - foobar_19[0:-2, 1:-1]) / (2 * foobar_8) * (foobar_22[1:-1, 2:] - foobar_22[1:-1, 0:-2]) / (2 * foobar_7)) - ((foobar_22[2:, 1:-1] - foobar_22[0:-2, 1:-1]) / (2 * foobar_8)) ** 2)
    foobar_4[1:-1, -1] = foobar_17 * (1 / foobar_6 * ((foobar_19[1:-1, 0] - foobar_19[1:-1, -2]) / (2 * foobar_7) + (foobar_22[2:, -1] - foobar_22[0:-2, -1]) / (2 * foobar_8)) - ((foobar_19[1:-1, 0] - foobar_19[1:-1, -2]) / (2 * foobar_7)) ** 2 - 2 * ((foobar_19[2:, -1] - foobar_19[0:-2, -1]) / (2 * foobar_8) * (foobar_22[1:-1, 0] - foobar_22[1:-1, -2]) / (2 * foobar_7)) - ((foobar_22[2:, -1] - foobar_22[0:-2, -1]) / (2 * foobar_8)) ** 2)
    foobar_4[1:-1, 0] = foobar_17 * (1 / foobar_6 * ((foobar_19[1:-1, 1] - foobar_19[1:-1, -1]) / (2 * foobar_7) + (foobar_22[2:, 0] - foobar_22[0:-2, 0]) / (2 * foobar_8)) - ((foobar_19[1:-1, 1] - foobar_19[1:-1, -1]) / (2 * foobar_7)) ** 2 - 2 * ((foobar_19[2:, 0] - foobar_19[0:-2, 0]) / (2 * foobar_8) * (foobar_22[1:-1, 1] - foobar_22[1:-1, -1]) / (2 * foobar_7)) - ((foobar_22[2:, 0] - foobar_22[0:-2, 0]) / (2 * foobar_8)) ** 2)
    return foobar_4

def fn2(foobar_14, foobar_7, foobar_8):
    foobar_15 = numpy.empty_like(foobar_14)
    for foobar_16 in range(foobar_9):
        foobar_15 = foobar_14.copy()
        foobar_14[1:-1, 1:-1] = ((foobar_15[1:-1, 2:] + foobar_15[1:-1, 0:-2]) * foobar_8 ** 2 + (foobar_15[2:, 1:-1] + foobar_15[0:-2, 1:-1]) * foobar_7 ** 2) / (2 * (foobar_7 ** 2 + foobar_8 ** 2)) - foobar_7 ** 2 * foobar_8 ** 2 / (2 * (foobar_7 ** 2 + foobar_8 ** 2)) * foobar_4[1:-1, 1:-1]
        foobar_14[1:-1, -1] = ((foobar_15[1:-1, 0] + foobar_15[1:-1, -2]) * foobar_8 ** 2 + (foobar_15[2:, -1] + foobar_15[0:-2, -1]) * foobar_7 ** 2) / (2 * (foobar_7 ** 2 + foobar_8 ** 2)) - foobar_7 ** 2 * foobar_8 ** 2 / (2 * (foobar_7 ** 2 + foobar_8 ** 2)) * foobar_4[1:-1, -1]
        foobar_14[1:-1, 0] = ((foobar_15[1:-1, 1] + foobar_15[1:-1, -1]) * foobar_8 ** 2 + (foobar_15[2:, 0] + foobar_15[0:-2, 0]) * foobar_7 ** 2) / (2 * (foobar_7 ** 2 + foobar_8 ** 2)) - foobar_7 ** 2 * foobar_8 ** 2 / (2 * (foobar_7 ** 2 + foobar_8 ** 2)) * foobar_4[1:-1, 0]
        foobar_14[-1, :] = foobar_14[-2, :]
        foobar_14[0, :] = foobar_14[1, :]
    return foobar_14
foobar_12 = 41
foobar_13 = 41
foobar_10 = 10
foobar_9 = 50
foobar_5 = 1
foobar_7 = 2 / (foobar_12 - 1)
foobar_8 = 2 / (foobar_13 - 1)
foobar_24 = numpy.linspace(0, 2, foobar_12)
foobar_25 = numpy.linspace(0, 2, foobar_13)
foobar_2, foobar_3 = numpy.meshgrid(foobar_24, foobar_25)
foobar_17 = 1
foobar_11 = 0.1
foobar_1 = 1
foobar_6 = 0.01
foobar_19 = numpy.zeros((foobar_13, foobar_12))
foobar_21 = numpy.zeros((foobar_13, foobar_12))
foobar_22 = numpy.zeros((foobar_13, foobar_12))
foobar_23 = numpy.zeros((foobar_13, foobar_12))
foobar_14 = numpy.ones((foobar_13, foobar_12))
foobar_15 = numpy.ones((foobar_13, foobar_12))
foobar_4 = numpy.zeros((foobar_13, foobar_12))
foobar_20 = 1
foobar_18 = 0
while foobar_20 > 0.001:
    foobar_21 = foobar_19.copy()
    foobar_23 = foobar_22.copy()
    foobar_4 = fn1(foobar_17, foobar_6, foobar_7, foobar_8, foobar_19, foobar_22)
    foobar_14 = fn2(foobar_14, foobar_7, foobar_8)
    foobar_19[1:-1, 1:-1] = foobar_21[1:-1, 1:-1] - foobar_21[1:-1, 1:-1] * foobar_6 / foobar_7 * (foobar_21[1:-1, 1:-1] - foobar_21[1:-1, 0:-2]) - foobar_23[1:-1, 1:-1] * foobar_6 / foobar_8 * (foobar_21[1:-1, 1:-1] - foobar_21[0:-2, 1:-1]) - foobar_6 / (2 * foobar_17 * foobar_7) * (foobar_14[1:-1, 2:] - foobar_14[1:-1, 0:-2]) + foobar_11 * (foobar_6 / foobar_7 ** 2 * (foobar_21[1:-1, 2:] - 2 * foobar_21[1:-1, 1:-1] + foobar_21[1:-1, 0:-2]) + foobar_6 / foobar_8 ** 2 * (foobar_21[2:, 1:-1] - 2 * foobar_21[1:-1, 1:-1] + foobar_21[0:-2, 1:-1])) + foobar_1 * foobar_6
    foobar_22[1:-1, 1:-1] = foobar_23[1:-1, 1:-1] - foobar_21[1:-1, 1:-1] * foobar_6 / foobar_7 * (foobar_23[1:-1, 1:-1] - foobar_23[1:-1, 0:-2]) - foobar_23[1:-1, 1:-1] * foobar_6 / foobar_8 * (foobar_23[1:-1, 1:-1] - foobar_23[0:-2, 1:-1]) - foobar_6 / (2 * foobar_17 * foobar_8) * (foobar_14[2:, 1:-1] - foobar_14[0:-2, 1:-1]) + foobar_11 * (foobar_6 / foobar_7 ** 2 * (foobar_23[1:-1, 2:] - 2 * foobar_23[1:-1, 1:-1] + foobar_23[1:-1, 0:-2]) + foobar_6 / foobar_8 ** 2 * (foobar_23[2:, 1:-1] - 2 * foobar_23[1:-1, 1:-1] + foobar_23[0:-2, 1:-1]))
    foobar_19[1:-1, -1] = foobar_21[1:-1, -1] - foobar_21[1:-1, -1] * foobar_6 / foobar_7 * (foobar_21[1:-1, -1] - foobar_21[1:-1, -2]) - foobar_23[1:-1, -1] * foobar_6 / foobar_8 * (foobar_21[1:-1, -1] - foobar_21[0:-2, -1]) - foobar_6 / (2 * foobar_17 * foobar_7) * (foobar_14[1:-1, 0] - foobar_14[1:-1, -2]) + foobar_11 * (foobar_6 / foobar_7 ** 2 * (foobar_21[1:-1, 0] - 2 * foobar_21[1:-1, -1] + foobar_21[1:-1, -2]) + foobar_6 / foobar_8 ** 2 * (foobar_21[2:, -1] - 2 * foobar_21[1:-1, -1] + foobar_21[0:-2, -1])) + foobar_1 * foobar_6
    foobar_19[1:-1, 0] = foobar_21[1:-1, 0] - foobar_21[1:-1, 0] * foobar_6 / foobar_7 * (foobar_21[1:-1, 0] - foobar_21[1:-1, -1]) - foobar_23[1:-1, 0] * foobar_6 / foobar_8 * (foobar_21[1:-1, 0] - foobar_21[0:-2, 0]) - foobar_6 / (2 * foobar_17 * foobar_7) * (foobar_14[1:-1, 1] - foobar_14[1:-1, -1]) + foobar_11 * (foobar_6 / foobar_7 ** 2 * (foobar_21[1:-1, 1] - 2 * foobar_21[1:-1, 0] + foobar_21[1:-1, -1]) + foobar_6 / foobar_8 ** 2 * (foobar_21[2:, 0] - 2 * foobar_21[1:-1, 0] + foobar_21[0:-2, 0])) + foobar_1 * foobar_6
    foobar_22[1:-1, -1] = foobar_23[1:-1, -1] - foobar_21[1:-1, -1] * foobar_6 / foobar_7 * (foobar_23[1:-1, -1] - foobar_23[1:-1, -2]) - foobar_23[1:-1, -1] * foobar_6 / foobar_8 * (foobar_23[1:-1, -1] - foobar_23[0:-2, -1]) - foobar_6 / (2 * foobar_17 * foobar_8) * (foobar_14[2:, -1] - foobar_14[0:-2, -1]) + foobar_11 * (foobar_6 / foobar_7 ** 2 * (foobar_23[1:-1, 0] - 2 * foobar_23[1:-1, -1] + foobar_23[1:-1, -2]) + foobar_6 / foobar_8 ** 2 * (foobar_23[2:, -1] - 2 * foobar_23[1:-1, -1] + foobar_23[0:-2, -1]))
    foobar_22[1:-1, 0] = foobar_23[1:-1, 0] - foobar_21[1:-1, 0] * foobar_6 / foobar_7 * (foobar_23[1:-1, 0] - foobar_23[1:-1, -1]) - foobar_23[1:-1, 0] * foobar_6 / foobar_8 * (foobar_23[1:-1, 0] - foobar_23[0:-2, 0]) - foobar_6 / (2 * foobar_17 * foobar_8) * (foobar_14[2:, 0] - foobar_14[0:-2, 0]) + foobar_11 * (foobar_6 / foobar_7 ** 2 * (foobar_23[1:-1, 1] - 2 * foobar_23[1:-1, 0] + foobar_23[1:-1, -1]) + foobar_6 / foobar_8 ** 2 * (foobar_23[2:, 0] - 2 * foobar_23[1:-1, 0] + foobar_23[0:-2, 0]))
    foobar_19[0, :] = 0
    foobar_19[-1, :] = 0
    foobar_22[0, :] = 0
    foobar_22[-1, :] = 0
    foobar_20 = (numpy.sum(foobar_19) - numpy.sum(foobar_21)) / numpy.sum(foobar_19)
    foobar_18 += 1