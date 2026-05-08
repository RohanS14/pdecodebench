from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax, random
import tqdm
import numpy as np
from jax_cfd.base import grids, initial_conditions
# Initial Condition
def rfft_mesh(Nx, Ny):
    kx = jnp.fft.fftfreq(Nx)[:, None] * Nx / (2 * jnp.pi) * jnp.ones((1, Ny // 2 + 1))
    ky = jnp.fft.rfftfreq(Ny)[None, :] * Ny / (2 * jnp.pi) * jnp.ones((Nx, 1))
    return kx, ky
# Spatial and Temporal Mesh Construction
def brick_wall_filter_2d(Nx, Ny):
    filter_ = jnp.zeros((Nx, Ny // 2 + 1))
    filter_ = filter_.at[:int(2 / 3 * Nx) // 2,
                         :int(2 / 3 * (Ny // 2 + 1))].set(1)
    filter_ = filter_.at[-int(2 / 3 * Nx) // 2:,
                         :int(2 / 3 * (Ny // 2 + 1))].set(1)
    return filter_
# Velocity array for calculation (via finite elements method)
def vorticity_to_velocity(vorticity_hat, kx, ky):
    two_pi_i = 2j * jnp.pi
    laplace = two_pi_i**2 * (kx**2 + ky**2)
    laplace = laplace.at[0, 0].set(1.0)
    psi_hat = -vorticity_hat / laplace
    vx_hat = two_pi_i * ky * psi_hat
    vy_hat = -two_pi_i * kx * psi_hat
    return vx_hat, vy_hat
# Loop running the process of integrating the system over the space and time mesh
def filtered_vorticity_ic(key, Nx, Ny, max_velocity, peak_wavenumber=4):
    grid = grids.Grid((Nx, Ny), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    v0 = initial_conditions.filtered_velocity_field(
        key, grid, maximum_velocity=max_velocity, peak_wavenumber=peak_wavenumber
    )
    vx, vy = v0[0].data, v0[1].data
    kx, ky = rfft_mesh(Nx, Ny)
    vx_hat = jnp.fft.rfftn(vx)
    vy_hat = jnp.fft.rfftn(vy)
    omega_hat = 2j * jnp.pi * (kx * vy_hat - ky * vx_hat)
    return jnp.fft.irfftn(omega_hat, s=(Nx, Ny))
# Process loop (on time mesh)
def explicit_terms(vorticity_hat, kx, ky, filter_, Nx, Ny):
    vx_hat, vy_hat = vorticity_to_velocity(vorticity_hat, kx, ky)
    vx = jnp.fft.irfftn(vx_hat, s=(Nx, Ny))
    vy = jnp.fft.irfftn(vy_hat, s=(Nx, Ny))
    grad_x_hat = 2j * jnp.pi * kx * vorticity_hat
    grad_y_hat = 2j * jnp.pi * ky * vorticity_hat
    grad_x = jnp.fft.irfftn(grad_x_hat, s=(Nx, Ny))
    grad_y = jnp.fft.irfftn(grad_y_hat, s=(Nx, Ny))
    advection = -(grad_x * vx + grad_y * vy)
    return jnp.fft.rfftn(advection) * filter_
# Initial Condition
def cn_rk4_step(vorticity_hat, kx, ky, filter_,
                Nx, Ny, linear_term, dt):
    alphas = [0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748, 1]
    betas  = [0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257]
    gammas = [0.1496590219993, 0.3792103129999, 0.8229550293869, 0.6994504559488, 0.1530572479681]
    def implicit_solve(omega_hat, mu):
        return omega_hat / (1.0 - mu * linear_term)
    u = vorticity_hat
    h = jnp.zeros_like(vorticity_hat)
    for i in range(5):
        h = explicit_terms(u, kx, ky, filter_, Nx, Ny) + betas[i] * h
        mu = 0.5 * dt * (alphas[i + 1] - alphas[i])
        u = implicit_solve(u + gammas[i] * dt * h + mu * linear_term * u, mu)
    return u
# Spatial and Temporal Mesh Construction
@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 9))
def run_ns2d(key,
             Nx, Ny,
             t_pts,
             t_eval,
             viscosity,
             drag,
             max_velocity,
             fixed_ic=None,
             target_N=None):
    t_eval_arr = jnp.array(t_eval)
    dt = t_eval_arr[1] - t_eval_arr[0]
    n_steps = len(t_eval_arr) - 1
    kx, ky = rfft_mesh(Nx, Ny)
    filter_ = brick_wall_filter_2d(Nx, Ny)
    laplace = (2j * jnp.pi)**2 * (kx**2 + ky**2)
    linear_term = viscosity * laplace - drag
    key, ic_key = random.split(key)
    omega0 = (
        fixed_ic if fixed_ic is not None
        else filtered_vorticity_ic(ic_key, Nx, Ny, max_velocity)
    )
    omega0_hat = jnp.fft.rfftn(omega0)
    save_every = n_steps // (t_pts - 1)
    def outer_body(carry, _):
        omega_hat, key = carry
        def inner_body(carry, _):
            omega_hat, key = carry
            key, _ = random.split(key)
            omega_hat_new = cn_rk4_step(
                omega_hat, kx, ky, filter_,
                Nx, Ny, linear_term, dt
            )
            return (omega_hat_new, key), None
        (omega_hat_new, key), _ = lax.scan(
            inner_body, (omega_hat, key), None, length=save_every
        )
        return (omega_hat_new, key), omega_hat_new
    (_, _), traj_hat = lax.scan(
        outer_body, (omega0_hat, key), None, length=t_pts - 1
    )
    traj_hat = jnp.concatenate([omega0_hat[None], traj_hat], axis=0)
    if target_N is not None and target_N != Nx:
        T = target_N
        def downsample(oh):
            oh_small = jnp.concatenate(
                [oh[:T // 2, :T // 2 + 1],
                 oh[-T // 2:, :T // 2 + 1]], axis=0)
            return jnp.fft.irfftn(oh_small, s=(T, T)) * (T / Nx) ** 2
        traj = jax.vmap(downsample)(traj_hat)
    else:
        traj = jax.vmap(lambda oh: jnp.fft.irfftn(oh, s=(Nx, Ny)))(traj_hat)
    return traj
def get_ns2d(
    n_samples,
    t_pts,
    Nx=64,
    Ny=64,
    key=random.PRNGKey(0),
    dt=0.01,
    T_end=10.0,
    viscosity=1e-3,
    drag=0.1,
    max_velocity=3.0,
    batch_size=32,
    fixed_ic=None,
    target_N=None,
):
    t_eval = tuple(np.arange(0.0, T_end + dt, dt).tolist())
    keys = random.split(key, n_samples)
    ns_fn = partial(
        run_ns2d,
        Nx=Nx,
        Ny=Ny,
        t_pts=t_pts,
        t_eval=t_eval,
        viscosity=viscosity,
        drag=drag,
        max_velocity=max_velocity,
        fixed_ic=fixed_ic,
        target_N=target_N,
    )
    out_shape = (
        n_samples,
        t_pts,
        target_N if target_N is not None else Nx,
        target_N if target_N is not None else Ny,
    )
    dataset = np.zeros(out_shape, dtype=np.float32)
    for i in tqdm.tqdm(range(0, n_samples, batch_size)):
        batch_keys = keys[i:i + batch_size]
        batch = vmap(ns_fn)(batch_keys)
        dataset[i:i + len(batch_keys)] = np.array(batch)
    return dataset[..., None]
key = jax.random.PRNGKey(0)
dataset = get_ns2d(
    n_samples=2,
    t_pts=100,
    Nx=64,
    Ny=64,
    key=key,
    dt=0.001,
    T_end=1,
    viscosity=1e-3,
    drag=0.1,
    max_velocity=7.0,
    batch_size=16,
    target_N=64,
)
