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

@partial(jax.jit, static_argnums=(-3, -2))
def double_well_fn(init, control, ts, dt, params, k, step_fn, key):
	alpha, beta= params
	sigma = jnp.sqrt(2/beta)

	
	def rhs(t, state):
		x, y = state
		dxdt = 4*(x*x-1)*x
		dydt = 2*alpha*y
		return jnp.array([-dxdt, -dydt])
	
	fine_noise = jnp.sqrt(dt)*jax.random.normal(key, shape=(control.shape[0],2))
	fine_noise = fine_noise.at[0, :].set(0.0)

	f = partial(step_fn, dt=dt, rhs=rhs, sigma=sigma)

	def scan_step_fn(state, inputs): 
		state_array = jnp.array(state)
		t, noise = inputs
		next_state_array = f(state_array, t, noise=noise)  # Call with explicit noise
		next_state = (next_state_array[0], next_state_array[1])  # Turn it back into a tuple
		return next_state, next_state
	

	# traj is solution with fine time step dt
	_, fine_traj = jax.lax.scan(scan_step_fn, init, (control, fine_noise))

	large_step_control = control[::k]
	fine_noise_reshaped = fine_noise[:(control.shape[0] // k) * k].reshape(-1, k, 2)  # shape (num_large_steps, k)
	large_noise = fine_noise_reshaped.sum(axis=1)

	f_large = partial(step_fn, dt=k*dt, rhs=rhs, sigma=sigma)
	
	def large_step_scan_fn(state, inputs):
		state_array = jnp.array(state)
		t, noise = inputs
		next_state_array= f_large(state_array, t, noise=noise)
		next_state = (next_state_array[0], next_state_array[1])  # Turn it back into a tuple
		return next_state, next_state
	
	(_, large_traj) = jax.lax.scan(large_step_scan_fn, init, (large_step_control, large_noise))

	selected_times = ts[::k]
	
	large_states = jnp.stack([large_traj[0], large_traj[1]], axis = 0)

	return selected_times, large_states, large_noise

@partial(jax.jit, static_argnums=(-3, -2))
def coupled_doublewell_fn(init, control, ts, dt, params, k, step_fn, key):
	alpha, beta= params
	sigma = jnp.sqrt(2/beta)

	
	def rhs(t, state):
		x, y = state
		dxdt = 4*(x*x-1)*x + alpha * y
		dydt = y + alpha*x
		return jnp.array([-dxdt, -dydt])
	
	fine_noise = jnp.sqrt(dt)*jax.random.normal(key, shape=(control.shape[0],2))
	fine_noise = fine_noise.at[0, :].set(0.0)

	f = partial(step_fn, dt=dt, rhs=rhs, sigma=sigma)

	def scan_step_fn(state, inputs): 
		state_array = jnp.array(state)
		t, noise = inputs
		next_state_array = f(state_array, t, noise=noise)  # Call with explicit noise
		next_state = (next_state_array[0], next_state_array[1])  # Turn it back into a tuple
		return next_state, next_state
	

	# traj is solution with fine time step dt
	_, fine_traj = jax.lax.scan(scan_step_fn, init, (control, fine_noise))

	large_step_control = control[::k]
	fine_noise_reshaped = fine_noise[:(control.shape[0] // k) * k].reshape(-1, k, 2)  # shape (num_large_steps, k)
	large_noise = fine_noise_reshaped.sum(axis=1)

	f_large = partial(step_fn, dt=k*dt, rhs=rhs, sigma=sigma)
	
	def large_step_scan_fn(state, inputs):
		state_array = jnp.array(state)
		t, noise = inputs
		next_state_array= f_large(state_array, t, noise=noise)
		next_state = (next_state_array[0], next_state_array[1])  # Turn it back into a tuple
		return next_state, next_state
	
	(_, large_traj) = jax.lax.scan(large_step_scan_fn, init, (large_step_control, large_noise))

	selected_times = ts[::k]
	
	large_states = jnp.stack([large_traj[0], large_traj[1]], axis = 0)

	return selected_times, large_states, large_noise


@partial(jax.jit, static_argnums=(-3, -2))
def mueller_overdamped_fn(init, control, ts, dt, params, k, step_fn, key):
	a_scale, b_scale, c_scale, D_scale, X_shift, Y_shift, beta = params
	sigma = jnp.sqrt(2/beta)

	
	def rhs(t, state):
		x, y = state  # unpack

		# constants (as jnp arrays)
		a = jnp.array([-1.0, -1.0, -6.5, 0.7])
		b = jnp.array([0.0, 0.0, 11.0, 0.6])
		c = jnp.array([-10.0, -10.0, -6.5, 0.7])
		D = jnp.array([-200.0, -100.0, -170.0, 15.0])
		X = jnp.array([1.0, 0.0, -0.5, -1.0])
		Y = jnp.array([0.0, 0.5, 1.5, 1.0])

		a = a*a_scale
		b = b*b_scale
		c = c*c_scale
		D = D*D_scale
		X = X + X_shift
		Y = Y + Y_shift

		gamma = 9.0
		k = 5.0
		pitorch = jnp.pi  # you called it pitorch

		# Compute the fx terms
		fx = []
		for i in range(4):
			fx_i = D[i] * jnp.exp(
				a[i] * (x - X[i])**2 + 
				b[i] * (x - X[i]) * (y - Y[i]) + 
				c[i] * (y - Y[i])**2
			)
			fx.append(fx_i)
		fx = jnp.stack(fx)  # shape (4,)

		# Extra sinusoidal term (periodic forcing)
		# extra = gamma * jnp.sin(2 * k * pitorch * x) * jnp.sin(2 * k * pitorch * y)

		# extrapx = gamma * jnp.cos(2 * k * pitorch * x) * jnp.sin(2 * k * pitorch * y) * 2 * pitorch * k
		# extrapy = gamma * jnp.cos(2 * k * pitorch * y) * jnp.sin(2 * k * pitorch * x) * 2 * pitorch * k

		# Compute derivatives
		dVx = jnp.sum(fx * (2 * a * (x - X) + b * (y - Y)))
		dVy = jnp.sum(fx * (2 * c * (y - Y) + b * (x - X)))

		return jnp.array([-dVx, -dVy])
	
	fine_noise = jnp.sqrt(dt)*jax.random.normal(key, shape=(control.shape[0],2))
	fine_noise = fine_noise.at[0, :].set(0.0)

	f = partial(step_fn, dt=dt, rhs=rhs, sigma=sigma)

	def scan_step_fn(state, inputs): 
		state_array = jnp.array(state)
		t, noise = inputs
		next_state_array = f(state_array, t, noise=noise)  # Call with explicit noise
		next_state = (next_state_array[0], next_state_array[1])  # Turn it back into a tuple
		return next_state, next_state
	

	# traj is solution with fine time step dt
	_, fine_traj = jax.lax.scan(scan_step_fn, init, (control, fine_noise))

	large_step_control = control[::k]
	fine_noise_reshaped = fine_noise[:(control.shape[0] // k) * k].reshape(-1, k, 2)  # shape (num_large_steps, k)
	large_noise = fine_noise_reshaped.sum(axis=1)

	f_large = partial(step_fn, dt=k*dt, rhs=rhs, sigma=sigma)
	
	def large_step_scan_fn(state, inputs):
		state_array = jnp.array(state)
		t, noise = inputs
		next_state_array= f_large(state_array, t, noise=noise)
		next_state = (next_state_array[0], next_state_array[1])  # Turn it back into a tuple
		return next_state, next_state
	
	(_, large_traj) = jax.lax.scan(large_step_scan_fn, init, (large_step_control, large_noise))

	selected_times = ts[::k]
	
	large_states = jnp.stack([large_traj[0], large_traj[1]], axis = 0)

	return selected_times, large_states, large_noise


@partial(jax.jit, static_argnums=(-3, -2))
def duffing_langevin_fn(init, control, ts, dt, params, k, step_fn, key):
	delta, alpha, beta, gamma, omega, epsilon = params
	sigma = jnp.sqrt(epsilon)

	
	def rhs(t, state):
		x, v = state
		dxdt = v
		dvdt = gamma * jnp.cos(omega * t) - delta * v - alpha * x - beta * x ** 3 
		return jnp.array([-dxdt, -dvdt])
	
	fine_noise = jnp.sqrt(dt)*jax.random.normal(key, shape=(control.shape[0],2))
	fine_noise = fine_noise.at[0, :].set(0.0)

	f = partial(step_fn, dt=dt, rhs=rhs, sigma=sigma)

	def scan_step_fn(state, inputs): 
		state_array = jnp.array(state)
		t, noise = inputs
		next_state_array = f(state_array, t, noise=noise)  # Call with explicit noise
		next_state = (next_state_array[0], next_state_array[1])  # Turn it back into a tuple
		return next_state, next_state
	

	# traj is solution with fine time step dt
	_, fine_traj = jax.lax.scan(scan_step_fn, init, (control, fine_noise))

	large_step_control = control[::k]
	fine_noise_reshaped = fine_noise[:(control.shape[0] // k) * k].reshape(-1, k, 2)  # shape (num_large_steps, k)
	large_noise = fine_noise_reshaped.sum(axis=1)

	f_large = partial(step_fn, dt=k*dt, rhs=rhs, sigma=sigma)
	
	def large_step_scan_fn(state, inputs):
		state_array = jnp.array(state)
		t, noise = inputs
		next_state_array= f_large(state_array, t, noise=noise)
		next_state = (next_state_array[0], next_state_array[1])  # Turn it back into a tuple
		return next_state, next_state
	
	(_, large_traj) = jax.lax.scan(large_step_scan_fn, init, (large_step_control, large_noise))

	selected_times = ts[::k]
	
	large_states = jnp.stack([large_traj[0], large_traj[1]], axis = 0)

	return selected_times, large_states, large_noise

@partial(jax.jit, static_argnums=(-3, -2))
def perturbed_nonlinearoscillator(init, control, ts, dt, params, k, step_fn, key):
	T, sig = params

	
	def rhs(t, state):
		x1, x2 = state
		dx1dt = T*(x1 - x2 - x1*(x1*x1 + x2*x2))
		dx2dt = T*(x1 + x2 - x2*(x1*x1 + x2*x2))
		return jnp.array([dx1dt, dx2dt])
	
	def sigma(t, state):
		x1, x2 = state
		F1 = jnp.sqrt(T) * sig * x1 * x2
		F2 = jnp.sqrt(T) * sig * x2 * x2
		return jnp.array([F1, F2])
	
	fine_noise = jnp.sqrt(dt)*jax.random.normal(key, shape=(control.shape[0],2))
	fine_noise = fine_noise.at[0, :].set(0.0)

	f = partial(step_fn, dt=dt, rhs=rhs, sigma=sigma)

	def scan_step_fn(state, inputs): 
		state_array = jnp.array(state)
		t, noise = inputs
		next_state_array = f(state_array, t, noise=noise)  # Call with explicit noise
		next_state = (next_state_array[0], next_state_array[1])  # Turn it back into a tuple
		return next_state, next_state
	

	# traj is solution with fine time step dt
	_, fine_traj = jax.lax.scan(scan_step_fn, init, (control, fine_noise))

	large_step_control = control[::k]
	fine_noise_reshaped = fine_noise[:(control.shape[0] // k) * k].reshape(-1, k, 2)  # shape (num_large_steps, k)
	large_noise = fine_noise_reshaped.sum(axis=1)

	f_large = partial(step_fn, dt=k*dt, rhs=rhs, sigma=sigma)
	
	def large_step_scan_fn(state, inputs):
		state_array = jnp.array(state)
		t, noise = inputs
		next_state_array= f_large(state_array, t, noise=noise)
		next_state = (next_state_array[0], next_state_array[1])  # Turn it back into a tuple
		return next_state, next_state
	
	(_, large_traj) = jax.lax.scan(large_step_scan_fn, init, (large_step_control, large_noise))

	selected_times = ts[::k]
	
	large_states = jnp.stack([large_traj[0], large_traj[1]], axis = 0)

	return selected_times, large_states, large_noise


@partial(jax.jit, static_argnums=(-3, -2))
def periodic_nonlinearoscillator(init, control, ts, dt, params, k, step_fn, key):
	omega, Omega, sig = params

	
	def rhs(t, state):
		x1, x2 = state
		dx1dt = (2*jnp.pi)/omega * (-Omega * x2 + x1 * (1 + jnp.sqrt(x1*x1 + x2*x2) * (jnp.cos(2*t*jnp.pi) - 1)))
		dx2dt = (2*jnp.pi)/omega * (Omega * x2 + x1 * (1 + jnp.sqrt(x1*x1 + x2*x2) * (jnp.cos(2*t*jnp.pi) - 1)))
		return jnp.array([dx1dt, dx2dt])
	
	def sigma(t, state):
		x1, x2 = state
		F1 = sig * jnp.sqrt(2*jnp.pi/omega) * x1 * x2
		F2 = sig * jnp.sqrt(2*jnp.pi/omega) * x2 * x2
		return jnp.array([F1, F2])
	
	fine_noise = jnp.sqrt(dt)*jax.random.normal(key, shape=(control.shape[0],2))
	fine_noise = fine_noise.at[0, :].set(0.0)

	f = partial(step_fn, dt=dt, rhs=rhs, sigma=sigma)

	def scan_step_fn(state, inputs): 
		state_array = jnp.array(state)
		t, noise = inputs
		next_state_array = f(state_array, t, noise=noise)  # Call with explicit noise
		next_state = (next_state_array[0], next_state_array[1])  # Turn it back into a tuple
		return next_state, next_state
	

	# traj is solution with fine time step dt
	_, fine_traj = jax.lax.scan(scan_step_fn, init, (control, fine_noise))

	large_step_control = control[::k]
	fine_noise_reshaped = fine_noise[:(control.shape[0] // k) * k].reshape(-1, k, 2)  # shape (num_large_steps, k)
	large_noise = fine_noise_reshaped.sum(axis=1)

	f_large = partial(step_fn, dt=k*dt, rhs=rhs, sigma=sigma)
	
	def large_step_scan_fn(state, inputs):
		state_array = jnp.array(state)
		t, noise = inputs
		next_state_array= f_large(state_array, t, noise=noise)
		next_state = (next_state_array[0], next_state_array[1])  # Turn it back into a tuple
		return next_state, next_state
	
	(_, large_traj) = jax.lax.scan(large_step_scan_fn, init, (large_step_control, large_noise))

	selected_times = ts[::k]
	
	large_states = jnp.stack([large_traj[0], large_traj[1]], axis = 0)

	return selected_times, large_states, large_noise


double_well_batch_fn = jax.jit(
    jax.vmap(
        double_well_fn,
        in_axes=(0, 0, None, None, None, None, None, 0),  # last arg `key` is batched!
        out_axes=(0, 0, 0)
    ),
    static_argnums=(-3, -2)
)

coupled_doublewell_batch_fn = jax.jit(
    jax.vmap(
        coupled_doublewell_fn,
        in_axes=(0, 0, None, None, None, None, None, 0),  # last arg `key` is batched!
        out_axes=(0, 0, 0)
    ),
    static_argnums=(-3, -2)
)

mueller_overdamped_batch_fn = jax.jit(
    jax.vmap(
        mueller_overdamped_fn,
        in_axes=(0, 0, None, None, None, None, None, 0),  # last arg `key` is batched!
        out_axes=(0, 0, 0)
    ),
    static_argnums=(-3, -2)
)

duffing_langevin_batch_fn = jax.jit(
    jax.vmap(
        duffing_langevin_fn,
        in_axes=(0, 0, None, None, None, None, None, 0),  # last arg `key` is batched!
        out_axes=(0, 0, 0)
    ),
    static_argnums=(-3, -2)
)

perturbed_nonlinearoscillator_batch_fn = jax.jit(
    jax.vmap(
        perturbed_nonlinearoscillator,
        in_axes=(0, 0, None, None, None, None, None, 0),  # last arg `key` is batched!
        out_axes=(0, 0, 0)
    ),
    static_argnums=(-3, -2)
)

periodic_nonlinearoscillator_batch_fn = jax.jit(
    jax.vmap(
        periodic_nonlinearoscillator,
        in_axes=(0, 0, None, None, None, None, None, 0),  # last arg `key` is batched!
        out_axes=(0, 0, 0)
    ),
    static_argnums=(-3, -2)
)



@partial(jax.jit, static_argnames=('ode_batch_fn', 'length', 'num', 'k', 'N','eqn'))
def generate_one_dyn(key, ode_batch_fn, dt, length, num,  init_range, params,k,N = 1, eqn = "duffing"):
	'''
	generate data for dynamics
	@param 
		key: jax.random.PRNGKey
		ode_batch_fn: e.g. ode_auto_const_batch_fn, jitted function
		dt: float, time step
		length: int, length of time series
		num: int, number of samples
		k_sigma, k_l: float, kernel parameters
		init_range: tuple, range of initial values
		coeffs: tuple, coefficients of the dynamics, will be unpacked and passed to ode_batch_fn
		control: 2D array (num, length), control signal, if None, generate with Gaussian process
	@return
		ts: 2D array (num, length, 1), time series
		control: 2D array (num, length, 1), control signal
		traj: 2D array (num, length, 1), trajectory
	'''
	ts = jnp.arange(length) * dt
	ts_expand = einshape("i->ji", ts, j = num) # 100,50
	control = ts_expand

	key, subkey1, subkey2 = jax.random.split(key, num = 3)
	init_x = jax.random.uniform(subkey1, (num,), minval=init_range[0][0], maxval=init_range[0][1])
	if eqn == 'mueller_overdamped':
		print('equation is mueller')
		# init_y = -(3.0 / 2.0) * init_x + 0.75
		init_y = -(2.0/3.0) * init_x + (1.0/3.0)
	else:
		init_y = jax.random.uniform(subkey2, (num,), minval=init_range[1][0], maxval=init_range[1][1])
	init = (init_x, init_y)

	if N > 1:
		init_x = jnp.repeat(init_x, repeats=N, axis=0)   # Repeat x part
		init_y = jnp.repeat(init_y, repeats=N, axis=0)   # Repeat y part
		init = (init_x, init_y)  # Recreate the tuple
		control = jnp.repeat(control, repeats=N, axis=0)

	key, subkey2 = jax.random.split(key)
	keys = jax.random.split(subkey2, num * N)  # shape: (num * N, 2)

	if eqn == 'perturbed_nonlinearoscillator' or eqn == 'periodic_nonlinearoscillator':
		selected_times, selected_u, dW = ode_batch_fn(
			init, control, ts, dt, params, k, euler_maruyama_posdep, keys
		)
	else:
		selected_times, selected_u, dW = ode_batch_fn(
			init, control, ts, dt, params, k, euler_maruyama, keys
		)

	# e.g., traj (2,50), selected_times (2, 10), selected_u (2,9), errors (2,9)

	return selected_times[..., None], selected_u[..., None], dW



if __name__ == "__main__":
	from jax.config import config
	config.update('jax_enable_x64', True)
	import haiku as hk
	import matplotlib.pyplot as plt
	
	seed = 2
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	alpha = jax.random.uniform(next(rng), minval = 0, maxval = 1)
	beta = jax.random.uniform(next(rng), minval = 5, maxval = 20)

	# selected_times, selected_u, dW = generate_one_dyn(key = next(rng), ode_batch_fn = double_well_batch_fn
	# 						, dt=1e-5, length = 200, num =1,  init_range = [(-1,1),(-1,1)], params = [alpha,beta]
	# 						,k = 10)
	# selected_times, selected_u, dW = generate_one_dyn(key = next(rng), ode_batch_fn = coupled_doublewell_batch_fn
	# 						, dt=1e-5, length = 200, num =1,  init_range = [(-1,1),(-1,1)], params = [alpha,beta]
	# 						,k = 10)

	# a_scale = jax.random.uniform(next(rng), minval = 0.8, maxval = 1.2)
	# b_scale = jax.random.uniform(next(rng), minval = 0.8, maxval = 1.2)
	# c_scale = jax.random.uniform(next(rng), minval = 0.8, maxval = 1.2)
	# D_scale = jax.random.uniform(next(rng), minval = 0.7, maxval = 1.3)
	# X_shift = jax.random.uniform(next(rng), minval = -0.1, maxval = 0.1)
	# Y_shift = jax.random.uniform(next(rng), minval = -0.1, maxval = 0.1)
	# beta = jax.random.uniform(next(rng), minval = 0.05, maxval = 2.0)

	# print("a: {}, b: {}, c: {}, D: {}, X: {}, Y:{}, beta: {}".format(a_scale, b_scale, c_scale, D_scale, X_shift, Y_shift, beta))

	# selected_times, selected_u, dW = generate_one_dyn(key = next(rng), ode_batch_fn = mueller_overdamped_batch_fn
	# 						, dt=1e-5, length = 5000, num =1,  init_range = [(-1.,0.5),(-0.5,1.5)], 
	# 						params = [a_scale, b_scale, c_scale, D_scale, X_shift, Y_shift, beta],k = 100)
	
	# delta = jax.random.uniform(next(rng), minval = 0.05, maxval = 0.5)
	# alpha = jax.random.uniform(next(rng), minval = -1., maxval = 1.)
	# beta = jax.random.uniform(next(rng), minval = 1., maxval = 10.0)
	# gamma = jax.random.uniform(next(rng), minval = 0.1, maxval = 1.0)
	# omega = jax.random.uniform(next(rng), minval = 0.5, maxval = 6.0)
	# epsilon = jax.random.uniform(next(rng), minval = 0.01, maxval = 0.1)
	
	# selected_times, selected_u, dW = generate_one_dyn(key = next(rng), ode_batch_fn = duffing_langevin_batch_fn
	# 						, dt=1e-4, length = 20000, num =1,  init_range = [(-1,1),(-1,1)], 
	# 						params = [delta, alpha, beta, gamma, omega, epsilon],k = 100)

	T = jax.random.uniform(next(rng), minval=0.01, maxval=2 * jnp.pi)
	sig = jax.random.uniform(next(rng), minval = 0.01, maxval = 1.)

	print("T: {}, sigma: {}".format(T, sig))
	
	selected_times, selected_u, dW = generate_one_dyn(key = next(rng), ode_batch_fn = perturbed_nonlinearoscillator_batch_fn
							, dt=1e-4, length = 10000, num = 1,  init_range = [(-1, 1),(-1,1)], 
							params = [T, sig],k = 100, eqn = 'perturbed_nonlinearoscillator')


	# omega = jax.random.uniform(next(rng), minval=1, maxval=jnp.pi)
	# Omega = jax.random.uniform(next(rng), minval=0.1, maxval=jnp.pi)
	# sig = jax.random.uniform(next(rng), minval = 0.01, maxval = 1.)

	# print("omega: {}, Omega: {}, sigma: {}".format(omega, Omega, sig))
	
	# selected_times, selected_u, dW = generate_one_dyn(key = next(rng), ode_batch_fn = periodic_nonlinearoscillator_batch_fn
	# 						, dt=1e-5, length = 20000, num = 1,  init_range = [(-1, 1),(-1,1)], 
	# 						params = [omega, Omega, sig],k = 100, eqn = 'periodic_nonlinearoscillator')

	# test du/dt = u, with ground truth u = exp(t)
	selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
	print('selected_times',selected_times.shape)

	print('selected_sol',selected_u.shape)
	# print('noise: ', dW)

	x = selected_u[0, :, 0]
	v = selected_u[0, :, 1]
	# print('x',x)
	# print('v',v)
	# exit()

	# Time points
	time = jnp.arange(selected_u.shape[1])

	# Plotting
	plt.figure(figsize=(10, 6))
	plt.plot(time, x, label='1d')
	plt.plot(time, v, label='2d')
	# plt.plot(time, errors[0, :, 0], label='error 1d')
	plt.xlabel('Time')

	plt.legend()
	plt.grid(True)
	# plt.savefig('periodic_nonlinearoscillator.png')
	plt.savefig('perturbed_nonlinearoscillator.png')
