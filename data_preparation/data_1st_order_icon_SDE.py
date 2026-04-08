import jax
import jax.numpy as jnp
from collections import namedtuple
from functools import partial
from einshape import jax_einshape as einshape

# Updated Euler-Maruyama to take random noise directly
def euler_maruyama(y, t, dt, rhs, sigma, noise):
    y_next = y + dt * rhs(t, y) + sigma * noise
    return y_next

def euler_maruyama_posdep(y, t, dt, rhs, sigma, noise):
	y_next = y + dt * rhs(t, y) + sigma(t,y) * noise
	return y_next

def gbm_milstein_step(y, dt, sigma, mu,  noise):
    '''
    This is from Milstein method wikipedia for reference
    '''
    # mu = rhs(t, y) / y  
    
    return y + mu * y * dt + sigma * y * noise + 0.5 * sigma**2 * y * (noise**2 - dt)





# Geometric Brownian Motion 
#########################################
@partial(jax.jit, static_argnums=(-3, -2))
def geombrownian_motion_fn(init, control, ts, dt, params, k, step_fn, key):
    mu, sigma = params

    def rhs(t, y):
        return mu * y
    def sigma_fn(t, y):
        return sigma * y
    

    # Pre-generate random numbers for fine-grained simulation
    fine_noise = jnp.sqrt(dt)*jax.random.normal(key, shape=(control.shape[0],))
    fine_noise = fine_noise.at[0].set(0.0)

    f = partial(step_fn, dt=dt, rhs=rhs, sigma=sigma_fn)
    
    def scan_step_fn(state, inputs): 
        state_array = jnp.array(state)
        t, noise = inputs
        next_state_array = f(state_array, t, noise=noise)  # Call with explicit noise
        next_state = next_state_array # Turn it back into a tuple
        return next_state, next_state
    


    # Run fine-grained simulation using pre-generated noises
    (_, fine_traj) = jax.lax.scan(scan_step_fn, init, (control, fine_noise))

    # ------------- Large step simulation (also pre-generated noises) ----------------
    large_step_control = control[::k]
    fine_noise_reshaped = fine_noise[:(control.shape[0] // k) * k].reshape(-1, k)  # shape (num_large_steps, k)
    large_noise = fine_noise_reshaped.sum(axis=1)  # sum over k steps

    f_large = partial(step_fn, dt=k*dt, rhs=rhs, sigma=sigma_fn)
    
    def large_step_scan_fn(state, inputs):
        state_array = jnp.array(state)
        t, noise = inputs
        next_state_array= f_large(state_array, t, noise=noise)
        next_state = next_state_array  # Turn it back into a tuple
        return next_state, next_state

    (_, large_traj) = jax.lax.scan(large_step_scan_fn, init, (large_step_control, large_noise))
    
    selected_times = ts[::k]

    return selected_times, large_traj, large_noise



####################################3333333333

@partial(jax.jit, static_argnums=(-3, -2))
def ornstein_uhlenbeck_fn(init, control, ts, dt, params, k, step_fn, key):
    theta, mu, sigma = params
    # def rhs(t, y):
    #     return theta * (mu - y)
    rhs = lambda t, y: theta * (mu - y)

    # Pre-generate random numbers for fine-grained simulation
    fine_noise = jnp.sqrt(dt)*jax.random.normal(key, shape=(control.shape[0],))
    fine_noise = fine_noise.at[0].set(0.0)
    
    def scan_step_fn(carry, inputs):
        y = carry
        t, noise = inputs
        y_next = step_fn(y, t, dt=dt, rhs=rhs, sigma=sigma, noise=noise)
        return y_next, y_next

    # Run fine-grained simulation using pre-generated noises
    (_, fine_traj) = jax.lax.scan(scan_step_fn, init, (control, fine_noise))

    # ------------- Large step simulation (also pre-generated noises) ----------------
    large_step_control = control[::k]
    fine_noise_reshaped = fine_noise[:(control.shape[0] // k) * k].reshape(-1, k)  # shape (num_large_steps, k)
    large_noise = fine_noise_reshaped.sum(axis=1)  # sum over k steps

    def large_step_scan_fn(carry, inputs):
        y = carry
        t, noise = inputs
        y_next = step_fn(y, t, dt=k*dt, rhs=rhs, sigma=sigma, noise=noise)
        return y_next, y_next

    (_, large_traj) = jax.lax.scan(large_step_scan_fn, init, (large_step_control, large_noise))

    selected_times = ts[::k]

    return selected_times, large_traj, large_noise



# inhomogeneous Ornstein–Uhlenbeck
@partial(jax.jit, static_argnums=(-3, -2))
def inhomogeneous_ornsteinuhlenbeck_fn(init, control, ts, dt, params, k, step_fn, key):
    a, omega, theta, sigma = params

    def rhs(t, y):
        return a * jnp.cos(omega*t) - theta * y
    
    fine_noise = jnp.sqrt(dt)*jax.random.normal(key, shape=(control.shape[0],))
    fine_noise = fine_noise.at[0].set(0.0)
    
    def scan_step_fn(carry, inputs):
        y = carry
        t, noise = inputs
        y_next = step_fn(y, t, dt=dt, rhs=rhs, sigma=sigma, noise=noise)
        return y_next, y_next

    # Run fine-grained simulation using pre-generated noises
    (_, fine_traj) = jax.lax.scan(scan_step_fn, init, (control, fine_noise))

    # ------------- Large step simulation (also pre-generated noises) ----------------
    large_step_control = control[::k]
    fine_noise_reshaped = fine_noise[:(control.shape[0] // k) * k].reshape(-1, k)  # shape (num_large_steps, k)
    large_noise = fine_noise_reshaped.sum(axis=1)  # sum over k steps

    def large_step_scan_fn(carry, inputs):
        y = carry
        t, noise = inputs
        y_next = step_fn(y, t, dt=k*dt, rhs=rhs, sigma=sigma, noise=noise)
        return y_next, y_next

    (_, large_traj) = jax.lax.scan(large_step_scan_fn, init, (large_step_control, large_noise))

    selected_times = ts[::k]

    return selected_times, large_traj, large_noise


geombrownian_motion_batch_fn = jax.jit(
    jax.vmap(
        geombrownian_motion_fn,
        in_axes=(0, 0, None, None, None, None, None, 0),
        out_axes=(0, 0, 0)
    ),
    static_argnums=(-3, -2)
)

ornstein_uhlenbeck_batch_fn = jax.jit(
    jax.vmap(
        ornstein_uhlenbeck_fn,
        in_axes=(0, 0, None, None, None, None, None, 0),  # last arg `key` is batched!
        out_axes=(0, 0, 0)
    ),
    static_argnums=(-3, -2)
)

inhomogeneous_ornsteinuhlenbeck_batch_fn = jax.jit(
    jax.vmap(
        inhomogeneous_ornsteinuhlenbeck_fn,
        in_axes=(0, 0, None, None, None, None, None, 0),  # last arg `key` is batched!
        out_axes=(0, 0, 0)
    ),
    static_argnums=(-3, -2)
)


@partial(jax.jit, static_argnames=('ode_batch_fn', 'length', 'num', 'k', 'N','eqn'))
def generate_one_dyn(key, ode_batch_fn, dt, length, num, init_range, params, k, N=1, eqn = "ornstein_uhlenbeck"):
    """
    Generate data for OU dynamics and repeat simulation N times per initial condition.

    Parameters:
        key: jax.random.PRNGKey
        ode_batch_fn: batched integrator function (e.g., ornstein_uhlenbeck_batch_fn)
        dt: float, time step size
        length: int, time series length
        num: int, number of unique initial values
        init_range: tuple, value range for initial state
        params: tuple (theta, mu, sigma)
        k: int, step multiplier for coarse integration
        N: int, number of repeats per initial condition

    Returns:
        selected_times: shape (num*N, steps, 1)
        selected_u: shape (num*N, steps, 1)
        large_noise: shape (num*N, steps, 1)
    """
    # Time grid and control signals
    ts = jnp.arange(length) * dt                      # shape: (length,)
    ts_expand = einshape("i->ji", ts, j=num)          # shape: (num, length)
    control = ts_expand                               # shape: (num, length)

    # Sample initial values
    key, subkey1 = jax.random.split(key)
    init = jax.random.uniform(subkey1, shape=(num,), minval=init_range[0], maxval=init_range[1])  # shape: (num,)

    # Repeat initial conditions and controls N times
    if N > 1:
        init = jnp.repeat(init, repeats=N, axis=0)              # shape: (num * N,)
        control = jnp.repeat(control, repeats=N, axis=0)        # shape: (num * N, length)

    # Generate batch of random keys, one per simulation
    key, subkey2 = jax.random.split(key)
    keys = jax.random.split(subkey2, num * N)  # shape: (num * N, 2)

    if eqn == 'ornstein_uhlenbeck' or eqn == 'inhomogeneous_ornsteinuhlenbeck':
        selected_times, selected_u, dW = ode_batch_fn(
			init, control, ts, dt, params, k, euler_maruyama, keys
		)
    else:
        selected_times, selected_u, dW = ode_batch_fn(
			init, control, ts, dt, params, k, euler_maruyama_posdep, keys
		)

    return selected_times[..., None], selected_u[..., None], dW[...,None]


if __name__ == "__main__":
     
    # This is deprecated
	# from jax.config import config
	# config.update('jax_enable_x64', True)

    
    import jax
    jax.config.update("jax_enable_x64", True)


     
    import haiku as hk
    import matplotlib.pyplot as plt
	
    seed = 2
    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))

    theta = jax.random.uniform(next(rng), minval = 0.1, maxval = 0.5)
    mu = jax.random.uniform(next(rng), minval = 1, maxval = 5)


    # For GMB make mu smaller:
    mu = jax.random.uniform(next(rng), minval=0.05, maxval=0.15) # [0.05, 0.15] annually for stock prices, could also be negative in declining markets
    sigma = jax.random.uniform(next(rng), minval =0.01, maxval = 0.06) # [0.1, 0.6] annually for stocks, higher for more volatile assets
    print(f"GBM --   mu: {mu}, sigma: {sigma}")



    selected_times, selected_u, dW = generate_one_dyn(key = next(rng), ode_batch_fn = ornstein_uhlenbeck_batch_fn
							, dt=0.05, length = 200, num =1,  init_range = (100,200), params = [theta, mu, sigma]
							,k = 10)
    
    # k = 5

    # selected_times, selected_u, dW = generate_one_dyn(key=next(rng), ode_batch_fn=geombrownian_motion_batch_fn, 
    #                                                           dt=1/252, 
    #                                                           length=200, num=1, init_range=(100, 200), 
    #                                                           params=[mu, sigma], k=k, eqn = "geombrownian_motion")

    a = jax.random.uniform(next(rng), minval=0.5, maxval=2.0)
    omega = jax.random.uniform(next(rng), minval=jnp.pi, maxval=4*jnp.pi)
    theta = jax.random.uniform(next(rng), minval=0.5, maxval=2.0)
    sigma = jax.random.uniform(next(rng), minval=0.1, maxval=0.5)

    selected_times, selected_u, dW = generate_one_dyn(key = next(rng), ode_batch_fn = inhomogeneous_ornsteinuhlenbeck_batch_fn
							, dt=0.001, length = 200, num =1,  init_range = (100,200), params = [a, omega, theta, sigma]
							,k = 100,eqn = "inhomogeneous_ornsteinuhlenbeck")
    
    selected_u = selected_u[0,:]
    print('selected_times',selected_times.shape)

    print('selected_sol',selected_u.shape)
    print('noise: ', dW.shape)

    x = selected_u
    print('x',x)
    # exit()

    # Time points
    time = jnp.arange(selected_u.shape[0])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time, x, label='1d')
    plt.xlabel('Time')

    plt.legend()
    plt.grid(True)
    # plt.savefig('periodic_nonlinearoscillator.png')
    plt.savefig('inhomogeneous_ornsteinuhlenbeck.png')    

