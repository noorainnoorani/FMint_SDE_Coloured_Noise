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

def euler_maruyama_predprey2(y, t, dt, rhs, sigma, noise, gaussian):
	y_next = y + dt * rhs(t, y, gaussian) + sigma(t,y) * noise
	return y_next

@partial(jax.jit, static_argnums=(-3, -2))
def fluxgate_sensor_fn(init, control, ts, dt, params, k, step_fn, key):
	c, lambda_, epsilon, omega= params

	def rhs(t, state):
		x1, x2, x3, y1, y2, y3 = state
		dx1dt = -x1 + jnp.tanh(c*(x1 + lambda_ * x2 + y1))
		dx2dt = -x2 + jnp.tanh(c*(x2 + lambda_ * x3 + y2))
		dx3dt = -x3 + jnp.tanh(c*(x3 + lambda_ * x1 + y3))
		dy1dt = -omega*y1 
		dy2dt = -omega*y2 
		dy3dt = -omega*y3
		return jnp.array([dx1dt, dx2dt, dx3dt, dy1dt, dy2dt, dy3dt])
	
	fine_noise = jnp.sqrt(dt)*jax.random.normal(key, shape=(control.shape[0],6))
	fine_noise = fine_noise.at[0, :].set(0.0)
	fine_noise = fine_noise.at[:,:3].set(0.0) # first three rows columns are 0.

	f = partial(step_fn, dt=dt, rhs=rhs)

	def scan_step_fn(state, inputs): 
		state_array = jnp.array(state)
		t, noise = inputs
		next_state_array = f(state_array, t, sigma = omega * jnp.sqrt(epsilon), noise=noise)  # Call with explicit noise
		next_state = (next_state_array[0], next_state_array[1], next_state_array[2], next_state_array[3], next_state_array[4], next_state_array[5])  # Turn it back into a tuple
		return next_state, next_state
	

	# traj is solution with fine time step dt
	_, fine_traj = jax.lax.scan(scan_step_fn, init, (control, fine_noise))

	large_step_control = control[::k]
	fine_noise_reshaped = fine_noise[:(control.shape[0] // k) * k].reshape(-1, k, 6)  # shape (num_large_steps, k)
	large_noise = fine_noise_reshaped.sum(axis=1)

	f_large = partial(step_fn, dt=k*dt, rhs=rhs)
	
	def large_step_scan_fn(state, inputs):
		state_array = jnp.array(state)
		t, noise = inputs
		next_state_array= f_large(state_array, t, sigma = omega * jnp.sqrt(epsilon), noise=noise)
		next_state = (next_state_array[0], next_state_array[1], next_state_array[2], next_state_array[3], next_state_array[4], next_state_array[5])  # Turn it back into a tuple
		return next_state, next_state
	
	(_, large_traj) = jax.lax.scan(large_step_scan_fn, init, (large_step_control, large_noise))

	selected_times = ts[::k]
	
	large_states = jnp.stack([large_traj[0], large_traj[1], large_traj[2]], axis = 0)
	large_noise_return = jnp.stack([large_traj[3], large_traj[4], large_traj[5]], axis = 0).T

	# print('original noise shape: ', large_noise.shape)
	# print('new error shape: ', large_noise_return.shape)
	return selected_times, large_states, large_noise_return

@partial(jax.jit, static_argnums=(-3, -2))
def stochastic_lorenz_fn(init, control, ts, dt, params, k, step_fn, key):
	sigma, rho, beta, eta1, eta2, eta3 = params

	def rhs(t, state):
		x, y, z = state
		dxdt = sigma * (y - x)
		dydt = x * (rho - z) - y
		dzdt = x * y - beta * z
		return jnp.array([dxdt, dydt, dzdt])
	
	def noise_fn(t, state):
		return jnp.array([eta1, eta2, eta3])
	
	fine_noise = jnp.sqrt(dt)*jax.random.normal(key, shape=(control.shape[0],3))
	fine_noise = fine_noise.at[0, :].set(0.0)

	f = partial(step_fn, dt=dt, rhs=rhs, sigma=noise_fn)

	def scan_step_fn(state, inputs): 
		state_array = jnp.array(state)
		t, noise = inputs
		next_state_array = f(state_array, t, noise=noise)  # Call with explicit noise
		next_state = (next_state_array[0], next_state_array[1], next_state_array[2])  # Turn it back into a tuple
		return next_state, next_state
	

	# traj is solution with fine time step dt
	_, fine_traj = jax.lax.scan(scan_step_fn, init, (control, fine_noise))

	large_step_control = control[::k]
	fine_noise_reshaped = fine_noise[:(control.shape[0] // k) * k].reshape(-1, k, 3)  # shape (num_large_steps, k)
	large_noise = fine_noise_reshaped.sum(axis=1)

	f_large = partial(step_fn, dt=k*dt, rhs=rhs, sigma=noise_fn)
	
	def large_step_scan_fn(state, inputs):
		state_array = jnp.array(state)
		t, noise = inputs
		next_state_array= f_large(state_array, t, noise=noise)
		next_state = (next_state_array[0], next_state_array[1], next_state_array[2])  # Turn it back into a tuple
		return next_state, next_state
	
	(_, large_traj) = jax.lax.scan(large_step_scan_fn, init, (large_step_control, large_noise))

	selected_times = ts[::k]
	
	large_states = jnp.stack([large_traj[0], large_traj[1], large_traj[2]], axis = 0)

	return selected_times, large_states, large_noise


@partial(jax.jit, static_argnums=(-3, -2))
def predator_prey_fn(init, control, ts, dt, params, k, step_fn, key):
	r, a, s, b, K, g, D, v1, v2, sigma1, sigma2, sigma3 = params

	def rhs(t, state):
		x, y1, y2 = state
		dxdt = x*(r - a*x + s*y1 - b*y2)
		dydt = K*x*y2 - y1*(g*x + D + v1)
		dzdt = D*y1 - v2*y2
		return jnp.array([dxdt, dydt, dzdt])
	
	def noise_fn(t, state):
		x, y1, y2 = state
		eta1 = sigma1 * x
		eta2 = sigma2 * y1
		eta3 = sigma3 * y2
		return jnp.array([eta1, eta2, eta3])
	
	fine_noise = jnp.sqrt(dt)*jax.random.normal(key, shape=(control.shape[0],3))
	fine_noise = fine_noise.at[0, :].set(0.0)

	f = partial(step_fn, dt=dt, rhs=rhs, sigma=noise_fn)

	def scan_step_fn(state, inputs): 
		state_array = jnp.array(state)
		t, noise = inputs
		next_state_array = f(state_array, t, noise=noise)  # Call with explicit noise
		next_state = (next_state_array[0], next_state_array[1], next_state_array[2])  # Turn it back into a tuple
		return next_state, next_state
	

	# traj is solution with fine time step dt
	_, fine_traj = jax.lax.scan(scan_step_fn, init, (control, fine_noise))

	large_step_control = control[::k]
	fine_noise_reshaped = fine_noise[:(control.shape[0] // k) * k].reshape(-1, k, 3)  # shape (num_large_steps, k)
	large_noise = fine_noise_reshaped.sum(axis=1)

	f_large = partial(step_fn, dt=k*dt, rhs=rhs, sigma=noise_fn)
	
	def large_step_scan_fn(state, inputs):
		state_array = jnp.array(state)
		t, noise = inputs
		next_state_array= f_large(state_array, t, noise=noise)
		next_state = (next_state_array[0], next_state_array[1], next_state_array[2])  # Turn it back into a tuple
		return next_state, next_state
	
	(_, large_traj) = jax.lax.scan(large_step_scan_fn, init, (large_step_control, large_noise))

	selected_times = ts[::k]
	
	large_states = jnp.stack([large_traj[0], large_traj[1], large_traj[2]], axis = 0)

	return selected_times, large_states, large_noise

@partial(jax.jit, static_argnums=(-3, -2))
def predator_prey2_fn(init, control, ts, dt, params, k, step_fn, key):
	r, a, s, b, K, g, D, v1, v2, sigma1, sigma2, sigma3, sigma4 = params

	def rhs(t, state, gaussian):
		x, y1, y2 = state
		# Ensure gaussian is a scalar for proper broadcasting
		gaussian_scalar = jnp.squeeze(gaussian)
		K_ = K + sigma4 * gaussian_scalar
		dxdt = x*(r - a*x + s*y1 - b*y2)
		dydt = K_*x*y2 - y1*(g*x + D + v1)
		dzdt = D*y1 - v2*y2
		return jnp.array([dxdt, dydt, dzdt])
	
	def noise_fn(t, state):
		x, y1, y2 = state
		eta1 = sigma1 * x
		eta2 = sigma2 * y1
		eta3 = sigma3 * y2
		return jnp.array([eta1, eta2, eta3])
	
	fine_noise = jnp.sqrt(dt)*jax.random.normal(key, shape=(control.shape[0],3))
	gaussian = jax.random.normal(key, shape=(control.shape[0],1))
	fine_noise = fine_noise.at[0, :].set(0.0)

	f = partial(step_fn, dt=dt, rhs=rhs, sigma=noise_fn)

	def scan_step_fn(state, inputs): 
		state_array = jnp.array(state)
		t, noise, g = inputs
		next_state_array = f(state_array, t, noise=noise, gaussian=g)  # Call with explicit noise and gaussian
		next_state = (next_state_array[0], next_state_array[1], next_state_array[2])  # Turn it back into a tuple
		return next_state, next_state
	

	# traj is solution with fine time step dt
	_, fine_traj = jax.lax.scan(scan_step_fn, init, (control, fine_noise, gaussian))

	large_step_control = control[::k]
	fine_noise_reshaped = fine_noise[:(control.shape[0] // k) * k].reshape(-1, k, 3)  # shape (num_large_steps, k)
	gaussian_reshaped = gaussian[:(control.shape[0] // k) * k].reshape(-1, k, 1)  # shape (num_large_steps, k, 1)
	large_noise = fine_noise_reshaped.sum(axis=1)
	large_gaussian = gaussian_reshaped.sum(axis=1)  # Sum over k steps

	f_large = partial(step_fn, dt=k*dt, rhs=rhs, sigma=noise_fn)
	
	def large_step_scan_fn(state, inputs):
		state_array = jnp.array(state)
		t, noise, g = inputs
		next_state_array= f_large(state_array, t, noise=noise, gaussian=g)
		next_state = (next_state_array[0], next_state_array[1], next_state_array[2])  # Turn it back into a tuple
		return next_state, next_state
	
	(_, large_traj) = jax.lax.scan(large_step_scan_fn, init, (large_step_control, large_noise, large_gaussian))

	selected_times = ts[::k]
	
	large_states = jnp.stack([large_traj[0], large_traj[1], large_traj[2]], axis = 0)

	return selected_times, large_states, large_noise


fluxgate_sensor_batch_fn = jax.jit(
    jax.vmap(
        fluxgate_sensor_fn,
        in_axes=(0, 0, None, None, None, None, None, 0),  # last arg `key` is batched!
        out_axes=(0, 0, 0)
    ),
    static_argnums=(-3, -2)
)

stochastic_lorenz_batch_fn = jax.jit(
    jax.vmap(
        stochastic_lorenz_fn,
        in_axes=(0, 0, None, None, None, None, None, 0),  # last arg `key` is batched!
        out_axes=(0, 0, 0)
    ),
    static_argnums=(-3, -2)
)

predator_prey_batch_fn = jax.jit(
    jax.vmap(
        predator_prey_fn,
        in_axes=(0, 0, None, None, None, None, None, 0),  # last arg `key` is batched!
        out_axes=(0, 0, 0)
    ),
    static_argnums=(-3, -2)
)

predator_prey2_batch_fn = jax.jit(
    jax.vmap(
        predator_prey2_fn,
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

	key, subkey1, subkey2, subkey3, subkey4, subkey5 = jax.random.split(key, num = 6)
	init_x = jax.random.uniform(key, (num,), minval=init_range[0][0], maxval=init_range[0][1])
	init_y = jax.random.uniform(subkey1, (num,), minval=init_range[1][0], maxval=init_range[1][1])
	init_z = jax.random.uniform(subkey2, (num,), minval=init_range[2][0], maxval=init_range[2][1])
	if eqn == 'fluxgate_sensor':
		init_d4 = jax.random.uniform(subkey3, (num,), minval=init_range[3][0], maxval=init_range[3][1])
		init_d5 = jax.random.uniform(subkey4, (num,), minval=init_range[4][0], maxval=init_range[4][1])
		init_d6 = jax.random.uniform(subkey5, (num,), minval=init_range[5][0], maxval=init_range[5][1])
		init = (init_x, init_y, init_z, init_d4, init_d5, init_d6)
	else:
		init = (init_x, init_y, init_z)

	if N > 1:
		init_x = jnp.repeat(init_x, repeats=N, axis=0)   # Repeat x part
		init_y = jnp.repeat(init_y, repeats=N, axis=0)   # Repeat y part
		init_z = jnp.repeat(init_z, repeats=N, axis=0)
		if eqn == 'fluxgate_sensor':
			init_d4 = jnp.repeat(init_d4, repeats=N, axis=0)
			init_d5 = jnp.repeat(init_d5, repeats=N, axis=0)
			init_d6 = jnp.repeat(init_d6, repeats=N, axis=0)
			init = (init_x, init_y, init_z, init_d4, init_d5, init_d6)
		else:
			init = (init_x, init_y, init_z)  # Recreate the tuple
		control = jnp.repeat(control, repeats=N, axis=0)

	key, subkey2 = jax.random.split(key)
	keys = jax.random.split(subkey2, num * N)  # shape: (num * N, 2)

	if eqn == 'stochastic_lorenz' or eqn == 'predator_prey':
		selected_times, selected_u, dW = ode_batch_fn(
			init, control, ts, dt, params, k, euler_maruyama_posdep, keys
		)
	elif eqn == 'predator_prey2':
		selected_times, selected_u, dW = ode_batch_fn(
			init, control, ts, dt, params, k, euler_maruyama_predprey2, keys
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
	c = jax.random.uniform(next(rng), minval = 3, maxval = 5)
	lambda_ = jax.random.uniform(next(rng), minval = 0.5, maxval = 0.6)
	epsilon = jax.random.uniform(next(rng), minval = 0.1, maxval = 0.5)
	omega = 3.0
	
	
	# selected_times, selected_u, dW = generate_one_dyn(key = next(rng), ode_batch_fn = fluxgate_sensor_batch_fn
	# 						, dt=1e-4, length = 10000, num = 1,  init_range = [(-1, 1),(-1,1),(-1,1),(0,0.5), (0,0.5),(0,0.5)], 
	# 						params = [c, lambda_, epsilon, omega],k = 100, eqn='fluxgate_sensor')
	
	# rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	# c_k = jax.random.uniform(next(rng), (2,), minval=3, maxval=5)
	# lambda_k = jax.random.uniform(next(rng), (2,), minval=0.1, maxval=1.0)
	# epsilon_k = jax.random.uniform(next(rng), (2,), minval = 0.1, maxval = 0.5)
	# omega_k = jnp.full_like(epsilon_k, 3.0)
	# print('omega_k shape, epsilon_k shape: ', omega_k.shape, epsilon_k.shape)

	# all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	# for i, (c, lambda_, epsilon,omega) in enumerate(zip(c_k, lambda_k, epsilon_k, omega_k)):
	# 	ts_expand,selected_u, dW, errors = generate_one_dyn(key = next(rng), ode_batch_fn = fluxgate_sensor_batch_fn, 
	# 							dt = 1e-4, length = 10000, num = 2, init_range = [(-1, 1),(-1,1),(-1,1),(0,0.5), (0,0.5),(0,0.5)],
	# 							params = [c, lambda_, epsilon, omega], k = 100, 
	# 							N = 40, eqn = 'fluxgate_sensor')

	# 	cond = jnp.concatenate([ts_expand, dW], axis=-1)

	# 	selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
	# 	errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))

	# 	all_ts.append(cond)
	# 	all_ys.append(selected_u)
	# 	all_errors.append(errors)
	# 	all_params.append("{:.4f}_{:.4f}_{:.4f}".format(c, lambda_, epsilon))
	# 	all_eqn_captions.append(None)
	# sigma = jax.random.uniform(next(rng), minval = 3, maxval = 5)
	# rho = jax.random.uniform(next(rng), minval = 0.5, maxval = 0.6)
	# beta = jax.random.uniform(next(rng), minval = 0.1, maxval = 0.5)
	# eta1 = jax.random.uniform(next(rng), minval = 3, maxval = 5)
	# eta2 = jax.random.uniform(next(rng), minval = 0.5, maxval = 0.6)
	# eta3 = jax.random.uniform(next(rng), minval = 0.1, maxval = 0.5)
	
	# selected_times, selected_u, dW = generate_one_dyn(key = next(rng), ode_batch_fn = stochastic_lorenz_batch_fn
	# 						, dt=1e-5, length = 10000, num = 1,  init_range = [(-1, 1),(-1,1),(-1,1)], 
	# 						params = [sigma, rho, beta, eta1, eta2, eta3], k = 100, eqn='stochastic_lorenz')

	r = jax.random.uniform(next(rng), minval = 0.3, maxval = 0.5)
	a = jax.random.uniform(next(rng), minval = 0.01, maxval = 0.03)
	s = jax.random.uniform(next(rng), minval = 0.2, maxval = 0.45)
	b = jax.random.uniform(next(rng), minval = 0.3, maxval = 0.5)
	K = jax.random.uniform(next(rng), minval = 0.3, maxval = 0.5)
	g = jax.random.uniform(next(rng), minval = 0.4, maxval = 0.6)
	D = jax.random.uniform(next(rng), minval = 0.2, maxval = 0.5)
	v1 = jax.random.uniform(next(rng), minval = 0.1, maxval = 0.5)
	v2 = jax.random.uniform(next(rng), minval = 0.1, maxval = 0.5)
	sigma1 = jax.random.uniform(next(rng), minval = 0.01, maxval = 0.5)
	sigma2 = jax.random.uniform(next(rng), minval = 0.01, maxval = 0.5)
	sigma3 = jax.random.uniform(next(rng), minval = 0.01, maxval = 0.5)
	sigma4 = jax.random.uniform(next(rng), minval = 0.0, maxval = 0.005)
	
	
	# selected_times, selected_u, dW = generate_one_dyn(key = next(rng), ode_batch_fn = predator_prey_batch_fn
	# 						, dt=0.1, length = 10000, num = 1,  init_range = [(0, 0.8),(0,0.8),(0,0.8)], 
	# 						params = [r, a, s, b, K, g, D, v1, v2, sigma1, sigma2, sigma3], k = 100, eqn='predator_prey')

	selected_times, selected_u, dW = generate_one_dyn(key = next(rng), ode_batch_fn = predator_prey2_batch_fn
							, dt=0.1, length = 10000, num = 1,  init_range = [(0, 0.8),(0,0.8),(0,0.8)], 
							params = [r, a, s, b, K, g, D, v1, v2, sigma1, sigma2, sigma3, sigma4], k = 100, eqn='predator_prey2')

	# test du/dt = u, with ground truth u = exp(t)
	selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
	print('selected_times',selected_times.shape)

	print('selected_sol',selected_u.shape)
	# print('noise: ', dW)

	x = selected_u[0, :, 0]
	v = selected_u[0, :, 1]
	z = selected_u[0, :, 2]
	# print('x',x)
	# print('v',v)
	# exit()

	# Time points
	time = jnp.arange(selected_u.shape[1])

	# Plotting
	plt.figure(figsize=(10, 6))
	plt.plot(time, x, label='1d')
	plt.plot(time, v, label='2d')
	plt.plot(time, z, label='3d')
	# plt.plot(time, errors[0, :, 0], label='error 1d')
	plt.xlabel('Time')

	plt.legend()
	plt.grid(True)
	# plt.savefig('fluxgate_sensor.png')
	# plt.savefig('stochastic_lorenz.png')
	plt.savefig('predator_prey.png')
