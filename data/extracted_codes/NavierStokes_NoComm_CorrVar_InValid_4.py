from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax, random
import tqdm
import numpy as np
from jax_cfd.base import grids, initial_conditions

def fn1(Nx, Ny):
    foobar_25 = jnp.fft.fftfreq(Nx)[:, None] * Nx / (2 * jnp.pi) * jnp.ones((1, Ny // 2 + 1))
    foobar_26 = jnp.fft.rfftfreq(Ny)[None, :] * Ny / (2 * jnp.pi) * jnp.ones((Nx, 1))
    return (foobar_25, foobar_26)

def fn2(Nx, Ny):
    foobar_13 = jnp.zeros((Nx, Ny // 2 + 1))
    foobar_13 = foobar_13.at[:int(2 / 3 * Nx) // 2, :int(2 / 3 * (Ny // 2 + 1))].set(1)
    foobar_13 = foobar_13.at[-int(2 / 3 * Nx) // 2:, :int(2 / 3 * (Ny // 2 + 1))].set(1)
    return foobar_13

def fn3(foobar_49, foobar_25, foobar_26):
    foobar_46 = 2j * jnp.pi
    foobar_27 = foobar_46 ** 2 * (foobar_25 ** 2 + foobar_26 ** 2)
    foobar_27 = foobar_27.at[0, 0].set(1.0)
    foobar_41 = -foobar_49 / foobar_27
    foobar_51 = foobar_46 * foobar_26 * foobar_41
    foobar_53 = -foobar_46 * foobar_25 * foobar_41
    return (foobar_51, foobar_53)

def fn4(foobar_23, Nx, Ny, max_velocity, foobar_40=4):
    foobar_19 = grids.Grid((Nx, Ny), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    foobar_48 = initial_conditions.filtered_velocity_field(foobar_23, foobar_19, maximum_velocity=max_velocity, peak_wavenumber=foobar_40)
    foobar_50, foobar_52 = (foobar_48[0].data, foobar_48[1].data)
    foobar_25, foobar_26 = fn1(Nx, Ny)
    foobar_51 = jnp.fft.rfftn(foobar_50)
    foobar_53 = jnp.fft.rfftn(foobar_52)
    foobar_37 = 2j * jnp.pi * (foobar_25 * foobar_53 - foobar_26 * foobar_51)
    return jnp.fft.irfftn(foobar_37, s=(Nx, Ny))

def fn5(foobar_49, foobar_25, foobar_26, foobar_13, Nx, Ny):
    foobar_51, foobar_53 = fn3(foobar_49, foobar_25, foobar_26)
    foobar_50 = jnp.fft.irfftn(foobar_51, s=(Nx, Ny))
    foobar_52 = jnp.fft.irfftn(foobar_53, s=(Nx, Ny))
    foobar_16 = 2j * jnp.pi * foobar_25 * foobar_49
    foobar_18 = 2j * jnp.pi * foobar_26 * foobar_49
    foobar_15 = jnp.fft.irfftn(foobar_16, s=(Nx, Ny))
    foobar_17 = jnp.fft.irfftn(foobar_18, s=(Nx, Ny))
    foobar_4 = foobar_15 * foobar_50 + foobar_17 * foobar_52
    return jnp.fft.rfftn(foobar_4) * foobar_13

def fn6(foobar_49, foobar_25, foobar_26, foobar_13, Nx, Ny, foobar_28, foobar_12):
    foobar_5 = [0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748, 1]
    foobar_9 = [0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257]
    foobar_14 = [0.1496590219993, 0.3792103129999, 0.8229550293869, 0.6994504559488, 0.1530572479681]

    def fn7(foobar_37, foobar_29):
        return foobar_37 / (1.0 - foobar_29 * foobar_28)
    foobar_47 = foobar_49
    foobar_20 = jnp.zeros_like(foobar_49)
    for foobar_21 in range(5):
        foobar_20 = fn5(foobar_47, foobar_25, foobar_26, foobar_13, Nx, Ny) + foobar_9[foobar_21] * foobar_20
        foobar_29 = 0.5 * foobar_12 * (foobar_5[foobar_21 + 1] - foobar_5[foobar_21])
        foobar_47 = fn7(foobar_47 + foobar_14[foobar_21] * foobar_12 * foobar_20 + foobar_29 * foobar_28 * foobar_47, foobar_29)
    return foobar_47

@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 9))
def fn8(foobar_23, Nx, Ny, t_pts, t_eval, viscosity, drag, max_velocity, fixed_ic=None, target_N=None):
    foobar_43 = jnp.array(t_eval)
    foobar_12 = foobar_43[1] - foobar_43[0]
    foobar_31 = len(foobar_43) - 1
    foobar_25, foobar_26 = fn1(Nx, Ny)
    foobar_13 = fn2(Nx, Ny)
    foobar_27 = (2j * jnp.pi) ** 2 * (foobar_25 ** 2 + foobar_26 ** 2)
    foobar_28 = -viscosity * foobar_27 - drag
    foobar_23, foobar_22 = random.split(foobar_23)
    foobar_35 = fixed_ic if fixed_ic is not None else fn4(foobar_22, Nx, Ny, max_velocity)
    foobar_36 = jnp.fft.rfftn(foobar_35)
    foobar_42 = foobar_31 // (t_pts - 1)

    def fn9(foobar_10, foobar_3):
        foobar_37, foobar_23 = foobar_10

        def fn10(foobar_10, foobar_3):
            foobar_37, foobar_23 = foobar_10
            foobar_23, foobar_3 = random.split(foobar_23)
            foobar_38 = fn6(foobar_37, foobar_25, foobar_26, foobar_13, Nx, Ny, foobar_28, foobar_12)
            return ((foobar_38, foobar_23), None)
        (foobar_38, foobar_23), foobar_3 = lax.scan(fn10, (foobar_37, foobar_23), None, length=foobar_42)
        return ((foobar_38, foobar_23), foobar_38)
    (foobar_3, foobar_3), foobar_45 = lax.scan(fn9, (foobar_36, foobar_23), None, length=t_pts - 1)
    foobar_45 = jnp.concatenate([foobar_36[None], foobar_45], axis=0)
    if target_N is not None and target_N != Nx:
        foobar_1 = target_N

        def fn11(foobar_33):
            foobar_34 = jnp.concatenate([foobar_33[:foobar_1 // 2, :foobar_1 // 2 + 1], foobar_33[-foobar_1 // 2:, :foobar_1 // 2 + 1]], axis=0)
            return jnp.fft.irfftn(foobar_34, s=(foobar_1, foobar_1)) * (foobar_1 / Nx) ** 2
        foobar_44 = jax.vmap(fn11)(foobar_45)
    else:
        foobar_44 = jax.vmap(lambda foobar_33: jnp.fft.irfftn(foobar_33, s=(Nx, Ny)))(foobar_45)
    return foobar_44

def fn12(foobar_30, t_pts, Nx=64, Ny=64, foobar_23=random.PRNGKey(0), foobar_12=0.01, foobar_2=10.0, viscosity=0.001, drag=0.1, max_velocity=3.0, foobar_8=32, fixed_ic=None, target_N=None):
    t_eval = tuple(np.arange(0.0, foobar_2 + foobar_12, foobar_12).tolist())
    foobar_24 = random.split(foobar_23, foobar_30)
    foobar_32 = partial(fn8, Nx=Nx, Ny=Ny, t_pts=t_pts, t_eval=t_eval, viscosity=viscosity, drag=drag, max_velocity=max_velocity, fixed_ic=fixed_ic, target_N=target_N)
    foobar_39 = (foobar_30, t_pts, target_N if target_N is not None else Nx, target_N if target_N is not None else Ny)
    foobar_11 = np.zeros(foobar_39, dtype=np.float32)
    for foobar_21 in tqdm.tqdm(range(0, foobar_30, foobar_8)):
        foobar_7 = foobar_24[foobar_21:foobar_21 + foobar_8]
        foobar_6 = vmap(foobar_32)(foobar_7)
        foobar_11[foobar_21:foobar_21 + len(foobar_7)] = np.array(foobar_6)
    return foobar_11[..., None]
foobar_23 = jax.random.PRNGKey(0)
foobar_11 = fn12(2, 100, 64, 64, foobar_23, 0.001, 1, 0.001, 0.1, 7.0, 16, None, 64)