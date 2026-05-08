import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
foobar_12 = 41
foobar_13 = 41
foobar_10 = 500
foobar_9 = 50
foobar_4 = 1
foobar_6 = 2 / (foobar_12 - 1)
foobar_7 = 2 / (foobar_13 - 1)
foobar_22 = numpy.linspace(0, 2, foobar_12)
foobar_23 = numpy.linspace(0, 2, foobar_13)
foobar_1, foobar_2 = numpy.meshgrid(foobar_22, foobar_23)
foobar_17 = 1
foobar_11 = 0.1
foobar_5 = 0.001
foobar_18 = numpy.zeros((foobar_13, foobar_12))
foobar_20 = numpy.zeros((foobar_13, foobar_12))
foobar_14 = numpy.zeros((foobar_13, foobar_12))
foobar_3 = numpy.zeros((foobar_13, foobar_12))

def fn1(foobar_3, foobar_17, foobar_5, foobar_18, foobar_20, foobar_6, foobar_7):
    foobar_3[1:-1, 1:-1] = foobar_17 * (1 / foobar_5 * ((foobar_18[1:-1, 2:] - foobar_18[1:-1, 0:-2]) / (2 * foobar_6) + (foobar_20[2:, 1:-1] - foobar_20[0:-2, 1:-1]) / (2 * foobar_7)) - ((foobar_18[1:-1, 2:] - foobar_18[1:-1, 0:-2]) / (2 * foobar_6)) ** 2 - 2 * ((foobar_18[2:, 1:-1] - foobar_18[0:-2, 1:-1]) / (2 * foobar_7) * (foobar_20[1:-1, 2:] - foobar_20[1:-1, 0:-2]) / (2 * foobar_6)) - ((foobar_20[2:, 1:-1] - foobar_20[0:-2, 1:-1]) / (2 * foobar_7)) ** 2)
    return foobar_3

def fn2(foobar_14, foobar_6, foobar_7, foobar_3):
    foobar_15 = numpy.empty_like(foobar_14)
    foobar_15 = foobar_14.copy()
    for foobar_16 in range(foobar_9):
        foobar_15 = foobar_14.copy()
        foobar_14[1:-1, 1:-1] = ((foobar_15[1:-1, 2:] + foobar_15[1:-1, 0:-2]) * foobar_7 ** 2 + (foobar_15[2:, 1:-1] + foobar_15[0:-2, 1:-1]) * foobar_6 ** 2) / (2 * (foobar_6 ** 2 + foobar_7 ** 2)) - foobar_6 ** 2 * foobar_7 ** 2 / (2 * (foobar_6 ** 2 + foobar_7 ** 2)) * foobar_3[1:-1, 1:-1]
        foobar_14[:, -1] = foobar_14[:, -2]
        foobar_14[0, :] = foobar_14[1, :]
        foobar_14[:, 0] = foobar_14[:, 1]
        foobar_14[-1, :] = 0
    return foobar_14

def fn3(foobar_10, foobar_18, foobar_20, foobar_5, foobar_6, foobar_7, foobar_14, foobar_17, foobar_11):
    foobar_19 = numpy.empty_like(foobar_18)
    foobar_21 = numpy.empty_like(foobar_20)
    foobar_3 = numpy.zeros((foobar_13, foobar_12))
    for foobar_8 in range(foobar_10):
        foobar_19 = foobar_18.copy()
        foobar_21 = foobar_20.copy()
        foobar_3 = fn1(foobar_3, foobar_17, foobar_5, foobar_18, foobar_20, foobar_6, foobar_7)
        foobar_14 = fn2(foobar_14, foobar_6, foobar_7, foobar_3)
        foobar_18[1:-1, 1:-1] = foobar_19[1:-1, 1:-1] - foobar_19[1:-1, 1:-1] * foobar_5 / foobar_6 * (foobar_19[1:-1, 1:-1] - foobar_19[1:-1, 0:-2]) - foobar_21[1:-1, 1:-1] * foobar_5 / foobar_7 * (foobar_19[1:-1, 1:-1] - foobar_19[0:-2, 1:-1]) - foobar_5 / (2 * foobar_17 * foobar_6) * (foobar_14[1:-1, 2:] - foobar_14[1:-1, 0:-2]) + foobar_11 * (foobar_5 / foobar_6 ** 2 * (foobar_19[1:-1, 2:] - 2 * foobar_19[1:-1, 1:-1] + foobar_19[1:-1, 0:-2]) + foobar_5 / foobar_7 ** 2 * (foobar_19[2:, 1:-1] - 2 * foobar_19[1:-1, 1:-1] + foobar_19[0:-2, 1:-1]))
        foobar_20[1:-1, 1:-1] = foobar_21[1:-1, 1:-1] - foobar_19[1:-1, 1:-1] * foobar_5 / foobar_6 * (foobar_21[1:-1, 1:-1] - foobar_21[1:-1, 0:-2]) - foobar_21[1:-1, 1:-1] * foobar_5 / foobar_7 * (foobar_21[1:-1, 1:-1] - foobar_21[0:-2, 1:-1]) - foobar_5 / (2 * foobar_17 * foobar_7) * (foobar_14[2:, 1:-1] - foobar_14[0:-2, 1:-1]) + foobar_11 * (foobar_5 / foobar_6 ** 2 * (foobar_21[1:-1, 2:] - 2 * foobar_21[1:-1, 1:-1] + foobar_21[1:-1, 0:-2]) + foobar_5 / foobar_7 ** 2 * (foobar_21[2:, 1:-1] - 2 * foobar_21[1:-1, 1:-1] + foobar_21[0:-2, 1:-1]))
        foobar_18[0, :] = 0
        foobar_18[:, 0] = 0
        foobar_18[:, -1] = 0
        foobar_18[-1, :] = 1
        foobar_20[0, :] = 0
        foobar_20[-1, :] = 0
        foobar_20[:, 0] = 0
        foobar_20[:, -1] = 0
    return (foobar_18, foobar_20, foobar_14)
foobar_18 = numpy.zeros((foobar_13, foobar_12))
foobar_20 = numpy.zeros((foobar_13, foobar_12))
foobar_14 = numpy.zeros((foobar_13, foobar_12))
foobar_3 = numpy.zeros((foobar_13, foobar_12))
foobar_10 = 700
foobar_18, foobar_20, foobar_14 = fn3(foobar_10, foobar_18, foobar_20, foobar_5, foobar_6, foobar_7, foobar_14, foobar_17, foobar_11)