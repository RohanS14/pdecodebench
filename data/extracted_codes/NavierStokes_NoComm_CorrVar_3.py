from time import time
import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
foobar_31 = 0.000625
foobar_13 = 0.1
foobar_25 = 0.01
foobar_9 = 6
foobar_10 = [2 ** foobar_9, 2 ** foobar_9, 2 ** foobar_9]
foobar_7 = np.array([2 * np.pi, 4 * np.pi, 4 * np.pi], dtype=float)
foobar_1 = PFFT(MPI.COMM_WORLD, foobar_10, collapse=False)
foobar_2 = foobar_1
foobar_14 = newDistArray(foobar_1, False, rank=1, view=True)
foobar_15 = newDistArray(foobar_1, rank=1, view=True)
foobar_11 = newDistArray(foobar_1, False, view=True)
foobar_12 = newDistArray(foobar_1, view=True)
foobar_16 = newDistArray(foobar_1, rank=1, view=True)
foobar_17 = newDistArray(foobar_1, rank=1, view=True)
foobar_20 = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]
foobar_21 = [0.5, 0.5, 1.0]
foobar_24 = newDistArray(foobar_1, rank=1, view=True)
foobar_22 = newDistArray(foobar_1, False, rank=1, view=True)
foobar_18 = newDistArray(foobar_2, False, rank=1, view=True)
foobar_23 = newDistArray(foobar_2, False, rank=1, view=True)

def fn1(foobar_1, foobar_7):
    foobar_19 = np.ogrid[foobar_1.local_slice(False)]
    foobar_10 = foobar_1.global_shape()
    foobar_19 = [np.broadcast_to(foobar_39 * foobar_7[foobar_26] / foobar_10[foobar_26], foobar_1.shape(False)) for foobar_26, foobar_39 in enumerate(foobar_19)]
    return foobar_19

def fn2(foobar_1, foobar_7):
    foobar_34 = foobar_1.local_slice()
    foobar_10 = foobar_1.global_shape()
    foobar_28 = [np.fft.fftfreq(foobar_30, 1.0 / foobar_30).astype(int) for foobar_30 in foobar_10[:-1]]
    foobar_28.append(np.fft.rfftfreq(foobar_10[-1], 1.0 / foobar_10[-1]).astype(int))
    foobar_3 = [foobar_29[foobar_35] for foobar_29, foobar_35 in zip(foobar_28, foobar_34)]
    foobar_6 = np.meshgrid(*foobar_3, indexing='ij', sparse=True)
    foobar_8 = 2 * np.pi / foobar_7
    return [np.broadcast_to(foobar_28 * foobar_8[foobar_26], foobar_1.shape(True)) for foobar_26, foobar_28 in enumerate(foobar_6)]
foobar_19 = fn1(foobar_1, foobar_7)
foobar_3 = fn2(foobar_1, foobar_7)
foobar_3 = np.array(foobar_3).astype(float)
foobar_4 = np.sum(foobar_3 * foobar_3, 0, dtype=float)
foobar_5 = foobar_3.astype(float) / np.where(foobar_4 == 0, 1, foobar_4).astype(float)

def fn3(foobar_39, foobar_40, foobar_41):
    foobar_41[0] = foobar_2.forward(foobar_39[1] * foobar_40[2] - foobar_39[2] * foobar_40[1], foobar_41[0])
    foobar_41[1] = foobar_2.forward(foobar_39[2] * foobar_40[0] - foobar_39[0] * foobar_40[2], foobar_41[1])
    foobar_41[2] = foobar_2.forward(foobar_39[0] * foobar_40[1] - foobar_39[1] * foobar_40[0], foobar_41[2])
    return foobar_41

def fn4(foobar_39, foobar_41):
    foobar_41[2] = foobar_2.backward(1j * (foobar_3[0] * foobar_39[1] - foobar_3[1] * foobar_39[0]), foobar_41[2])
    foobar_41[1] = foobar_2.backward(1j * (foobar_3[2] * foobar_39[0] - foobar_3[0] * foobar_39[2]), foobar_41[1])
    foobar_41[0] = foobar_2.backward(1j * (foobar_3[1] * foobar_39[2] - foobar_3[2] * foobar_39[1]), foobar_41[0])
    return foobar_41

def fn5(foobar_32):
    for foobar_27 in range(3):
        foobar_18[foobar_27] = foobar_2.backward(foobar_15[foobar_27], foobar_18[foobar_27])
    foobar_23[:] = fn4(foobar_15, foobar_23)
    foobar_32 = fn3(foobar_18, foobar_23, foobar_32)
    foobar_12[:] = np.sum(foobar_32 * foobar_5, 0, out=foobar_12)
    foobar_32 -= foobar_12 * foobar_3
    foobar_32 -= foobar_31 * foobar_4 * foobar_15
    return foobar_32
foobar_14[0] = np.sin(foobar_19[0]) * np.cos(foobar_19[1]) * np.cos(foobar_19[2])
foobar_14[1] = -np.cos(foobar_19[0]) * np.sin(foobar_19[1]) * np.cos(foobar_19[2])
foobar_14[2] = 0
for foobar_26 in range(3):
    foobar_15[foobar_26] = foobar_1.forward(foobar_14[foobar_26], foobar_15[foobar_26])
foobar_36 = 0.0
foobar_38 = 0
foobar_37 = time()
while foobar_36 < foobar_13 - 1e-08:
    foobar_36 += foobar_25
    foobar_38 += 1
    foobar_17[:] = foobar_16[:] = foobar_15
    for foobar_33 in range(4):
        foobar_24 = fn5(foobar_24)
        if foobar_33 < 3:
            foobar_15[:] = foobar_16 + foobar_21[foobar_33] * foobar_25 * foobar_24
        foobar_17[:] += foobar_20[foobar_33] * foobar_25 * foobar_24
    foobar_15[:] = foobar_17[:]
    for foobar_26 in range(3):
        foobar_14[foobar_26] = foobar_1.backward(foobar_15[foobar_26], foobar_14[foobar_26])
foobar_28 = MPI.COMM_WORLD.reduce(np.sum(foobar_14 * foobar_14) / foobar_10[0] / foobar_10[1] / foobar_10[2] / 2)
if MPI.COMM_WORLD.Get_rank() == 0:
    print('Time = {}'.format(time() - foobar_37))
    assert round(float(foobar_28) - 0.124953117517, 7) == 0
foobar_1.destroy()