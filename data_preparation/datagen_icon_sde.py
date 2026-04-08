import numpy as np
import jax.numpy as jnp
import jax
from einshape import jax_einshape as einshape
import pickle
from functools import partial
import sys
sys.path.append('../')
import utils
from absl import app, flags, logging
import haiku as hk
import matplotlib.pyplot as plt
import time

import data_preparation.data_1st_order_icon_SDE as ode_1
import data_preparation.data_2nd_order_icon_SDE as ode_2
import data_preparation.data_3rd_order_icon_SDE as ode_3
import data_dynamics as dyn
import data_writetfrecord_icon_SDE as datawrite
import data_utils

def is_valid_batch(selected_u):
    # Check for any NaNs or infs in the batch
    def is_finite(x):
        return jnp.all(jnp.isfinite(x))

    return is_finite(selected_u)


def generate_valid_batch(rng, ode_batch_fn, dt, length, num, init_range, params, k, N, eqn, dim = '2'):
    max_tries = 1000  # To avoid infinite loop in case of persistent instability
    for attempt in range(max_tries):
        key = next(rng)
        if dim == '2':
            ts_expand, selected_u, dW = ode_2.generate_one_dyn(
                key=key,
                ode_batch_fn=ode_batch_fn,
                dt=dt,
                length=length,
                num=num,
                init_range=init_range,
                params=params,
                k=k,
                N=N,
                eqn=eqn)
        elif dim == '1':
            ts_expand, selected_u, dW = ode_1.generate_one_dyn(
                key=key,
                ode_batch_fn=ode_batch_fn,
                dt=dt,
                length=length,
                num=num,
                init_range=init_range,
                params=params,
                k=k,
                N=N,
                eqn=eqn
            )
        elif dim == '3':
            ts_expand, selected_u, dW = ode_3.generate_one_dyn(
                key=key,
                ode_batch_fn=ode_batch_fn,
                dt=dt,
                length=length,
                num=num,
                init_range=init_range,
                params=params,
                k=k,
                N=N,
                eqn=eqn
            )
        if is_valid_batch(selected_u):
            return ts_expand, selected_u, dW
        else:
            print(f"[Attempt {attempt+1}] Invalid batch detected, regenerating...")

    raise RuntimeError("Exceeded maximum number of retries while generating a valid batch.")



def generate_ornstein_uhlenbeck(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,num_repeat):
	'''dx/dt = theta*(mu - x)dt + sigma*dW'''
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	theta_k = jax.random.uniform(next(rng), (eqns,), minval = 0.1, maxval = 0.5)
	mu_k = jax.random.uniform(next(rng), (eqns,), minval = 1, maxval = 5)
	sigma_k = jax.random.uniform(next(rng), (eqns,), minval =0.1, maxval = 0.5)

	all_ts = []; all_ys = []; all_params = []; all_eqn_captions = []
	for i, (theta, mu, sigma) in enumerate(zip(theta_k, mu_k, sigma_k)):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, dW = ode_1.generate_one_dyn(key = next(rng), ode_batch_fn = ode_1.ornstein_uhlenbeck_batch_fn, 
                                    dt = dt, length = length, num = num, init_range = (50,100),
                                    params = [theta, mu, sigma], k = nv_step, N = num_repeat, eqn="ornstein_uhlenbeck")

			cond = jnp.concatenate([ts_expand, dW], axis=-1)

			all_ts.append(cond)
			all_ys.append(selected_u)
			all_params.append("{:.4f}_{:.4f}_{:.4f}".format(theta, mu, sigma))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_ICON_SDE_1st_order_tfrecord(name = name, eqn_type = "ornstein_uhlenbeck", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys,alpha=nv_step, repeat=num_repeat)
	
def generate_double_well(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,num_repeat):
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	alpha_k = jax.random.uniform(next(rng), (eqns,), minval = 0.1, maxval = 0.5)
	beta_k = jax.random.uniform(next(rng), (eqns,), minval = 5, maxval = 20)

	all_ts = []; all_ys = []; all_params = []; all_eqn_captions = []
	for i, (alpha,beta) in enumerate(zip(alpha_k, beta_k)):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, dW = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.double_well_batch_fn, 
									dt = dt, length = length, num = num, init_range = [(-1,1),(-1,1)],
									params = [alpha,beta], k = nv_step, N = num_repeat)
			# ts_expand (100, 19, 1)
			# dW (100,19,2)
			# selected_u (100, 19, 2)

			cond = jnp.concatenate([ts_expand, dW], axis=-1)

			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))

			all_ts.append(cond)
			all_ys.append(selected_u)
			all_params.append("{:.4f}_{:.4f}".format(alpha, beta))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_ICON_SDE_2nd_order_tfrecord(name = name, eqn_type = "double_well", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
			all_ts = all_ts, all_ys = all_ys,alpha=nv_step, repeat=num_repeat)
	
def generate_coupled_doublewell(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,num_repeat):
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	alpha_k = jax.random.uniform(next(rng), (eqns,), minval = 0.1, maxval = 0.5)
	beta_k = jax.random.uniform(next(rng), (eqns,), minval = 5, maxval = 20)

	all_ts = []; all_ys = []; all_params = []; all_eqn_captions = []
	for i, (alpha,beta) in enumerate(zip(alpha_k, beta_k)):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, dW = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.coupled_doublewell_batch_fn, 
									dt = dt, length = length, num = num, init_range = [(-1,1),(-1,1)],
									params = [alpha,beta], k = nv_step, N = num_repeat)
			# ts_expand (100, 19, 1)
			# dW (100,19,2)
			# selected_u (100, 19, 2)

			cond = jnp.concatenate([ts_expand, dW], axis=-1)

			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))

			all_ts.append(cond)
			all_ys.append(selected_u)
			all_params.append("{:.4f}_{:.4f}".format(alpha, beta))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_ICON_SDE_2nd_order_tfrecord(name = name, eqn_type = "coupled_doublewell", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
			all_ts = all_ts, all_ys = all_ys,alpha=nv_step, repeat=num_repeat)
	
def generate_mueller_overdamped(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,num_repeat):
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	a_scale_k = jax.random.uniform(next(rng), (eqns,), minval = 0.8, maxval = 1.2)
	b_scale_k = jax.random.uniform(next(rng), (eqns,), minval = 0.8, maxval = 1.2)
	c_scale_k = jax.random.uniform(next(rng), (eqns,), minval = 0.8, maxval = 1.2)
	D_scale_k = jax.random.uniform(next(rng), (eqns,), minval = 0.7, maxval = 1.3)
	X_shift_k = jax.random.uniform(next(rng), (eqns,), minval = -0.1, maxval = 0.1)
	Y_shift_k = jax.random.uniform(next(rng), (eqns,), minval = -0.1, maxval = 0.1)
	beta_k = jax.random.uniform(next(rng), (eqns,), minval = 0.05, maxval = 2)

	all_ts = []; all_ys = []; all_params = []; all_eqn_captions = []
	for i, (a_scale, b_scale, c_scale, D_scale, X_shift, Y_shift, beta) in enumerate(zip(a_scale_k, b_scale_k, c_scale_k, D_scale_k, X_shift_k, Y_shift_k, beta_k)):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, dW = generate_valid_batch(rng, ode_batch_fn = ode_2.mueller_overdamped_batch_fn, 
									dt = dt, length = length, num = num, init_range = [(-0.5,0.5),(-0.5,1.5)],
									params = [a_scale, b_scale, c_scale, D_scale, X_shift, Y_shift, beta], k = nv_step, 
									N = num_repeat, eqn = "mueller_overdamped", dim = '2')
			# ts_expand (100, 19, 1)
			# dW (100,19,2)
			# cond (100,19,3)
			# selected_u (100, 19, 2)

			cond = jnp.concatenate([ts_expand, dW], axis=-1)

			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))

			all_ts.append(cond)
			all_ys.append(selected_u)
			all_params.append("{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}".format(a_scale, b_scale, c_scale, D_scale, X_shift, Y_shift, beta))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_ICON_SDE_2nd_order_tfrecord(name = name, eqn_type = "mueller_overdamped", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
			all_ts = all_ts, all_ys = all_ys,alpha=nv_step, repeat=num_repeat)

def generate_duffing_langevin(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,num_repeat):
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	delta_k = jax.random.uniform(next(rng), (eqns,), minval = 0.05, maxval = 0.5)
	alpha_k = jax.random.uniform(next(rng), (eqns,), minval = -1., maxval = 1.)
	beta_k = jax.random.uniform(next(rng), (eqns,), minval = 1., maxval = 10.0)
	gamma_k = jax.random.uniform(next(rng), (eqns,), minval = 0.1, maxval = 1.0)
	omega_k = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 6.0)
	epsilon_k = jax.random.uniform(next(rng), (eqns,), minval = 0.01, maxval = 0.1)

	all_ts = []; all_ys = []; all_params = []; all_eqn_captions = []
	for i, (delta, alpha, beta, gamma, omega, epsilon) in enumerate(zip(delta_k, alpha_k, beta_k, gamma_k, omega_k, epsilon_k)):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, dW = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.duffing_langevin_batch_fn, 
									dt = dt, length = length, num = num, init_range = [(-1,1),(-1,1)],
									params = [delta, alpha, beta, gamma, omega, epsilon], k = nv_step, N = num_repeat)
			# ts_expand (100, 19, 1)
			# dW (100,19,2)
			# selected_u (100, 19, 2)

			cond = jnp.concatenate([ts_expand, dW], axis=-1)

			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))

			all_ts.append(cond)
			all_ys.append(selected_u)
			all_params.append("{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}".format(delta, alpha, beta, gamma, omega, epsilon))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_ICON_SDE_2nd_order_tfrecord(name = name, eqn_type = "duffing_langevin", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
			all_ts = all_ts, all_ys = all_ys,alpha=nv_step, repeat=num_repeat)
	
def generate_perturbed_nonlinearoscillator(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,num_repeat):
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	T_k = jax.random.uniform(next(rng), (eqns,), minval=0.01, maxval=2 * jnp.pi)
	sig_k = jax.random.uniform(next(rng), (eqns,), minval = 0.01, maxval = 1.)

	all_ts = []; all_ys = []; all_params = []; all_eqn_captions = []
	for i, (T, sigma) in enumerate(zip(T_k, sig_k)):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, dW = generate_valid_batch(rng, ode_batch_fn = ode_2.perturbed_nonlinearoscillator_batch_fn, 
									dt = dt, length = length, num = num, init_range = [(-1,1),(-1,1)],
									params = [T, sigma], k = nv_step, 
									N = num_repeat, eqn = 'perturbed_nonlinearoscillator', dim = '2')
			# ts_expand (100, 19, 1)
			# dW (100,19,2)
			# selected_u (100, 19, 2)

			cond = jnp.concatenate([ts_expand, dW], axis=-1)

			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))

			all_ts.append(cond)
			all_ys.append(selected_u)
			all_params.append("{:.4f}_{:.4f}".format(T, sigma))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_ICON_SDE_2nd_order_tfrecord(name = name, eqn_type = "perturbed_nonlinearoscillator", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
			all_ts = all_ts, all_ys = all_ys,alpha=nv_step, repeat=num_repeat)

def generate_periodic_nonlinearoscillator(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,num_repeat):
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	omega_k = jax.random.uniform(next(rng), (eqns,), minval=jnp.pi, maxval=2*jnp.pi)
	Omega_k = jax.random.uniform(next(rng), (eqns,), minval=0.1, maxval=jnp.pi)
	sig_k = jax.random.uniform(next(rng), (eqns,), minval = 0.01, maxval = 1.)

	all_ts = []; all_ys = []; all_params = []; all_eqn_captions = []
	for i, (omega, Omega, sig) in enumerate(zip(omega_k, Omega_k, sig_k)):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, dW = generate_valid_batch(rng, ode_batch_fn = ode_2.periodic_nonlinearoscillator_batch_fn, 
									dt = dt, length = length, num = num, init_range = [(-1,1),(-1,1)],
									params = [omega, Omega, sig], k = nv_step, 
									N = num_repeat, eqn = 'periodic_nonlinearoscillator', dim = '2')
			# ts_expand (100, 19, 1)
			# dW (100,19,2)
			# selected_u (100, 19, 2)

			cond = jnp.concatenate([ts_expand, dW], axis=-1)

			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))

			all_ts.append(cond)
			all_ys.append(selected_u)
			all_params.append("{:.4f}_{:.4f}_{:.4f}".format(omega, Omega, sig))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_ICON_SDE_2nd_order_tfrecord(name = name, eqn_type = "periodic_nonlinearoscillator", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
			all_ts = all_ts, all_ys = all_ys,alpha=nv_step, repeat=num_repeat)

def generate_GMB(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,num_repeat):
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	mu_k = jax.random.uniform(next(rng), (eqns,), minval = 0.01, maxval = 0.15)
	sigma_k = jax.random.uniform(next(rng), (eqns,), minval =0.01, maxval = 0.2)

	all_ts = []; all_ys = []; all_params = []; all_eqn_captions = []
	for i, (mu, sigma) in enumerate(zip(mu_k, sigma_k)):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, dW = generate_valid_batch(rng, ode_batch_fn = ode_1.geombrownian_motion_batch_fn, 
                                    dt = dt, length = length, num = num, init_range = (50,200),
                                    params = [mu, sigma], k = nv_step, N = num_repeat, eqn="geombrownian_motion", dim = '1')
			cond = jnp.concatenate([ts_expand, dW], axis=-1)
			all_ts.append(cond)
			all_ys.append(selected_u)
			all_params.append("{:.4f}_{:.4f}".format(mu, sigma))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_ICON_SDE_1st_order_tfrecord(name = name, eqn_type = "geombrownian_motion", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys,alpha=nv_step, repeat=num_repeat)

def generate_inhomogeneous_ornsteinuhlenbeck(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,num_repeat):
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	a_k = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 2.0)
	omega_k = jax.random.uniform(next(rng), (eqns,), minval=jnp.pi, maxval=4*jnp.pi)
	theta_k = jax.random.uniform(next(rng), (eqns,), minval=0.5, maxval=2.0)
	sigma_k = jax.random.uniform(next(rng), (eqns,), minval=0.1, maxval=0.5)

	all_ts = []; all_ys = []; all_params = []; all_eqn_captions = []
	for i, (a, omega, theta, sigma) in enumerate(zip(a_k, omega_k, theta_k, sigma_k)):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, dW = generate_valid_batch(rng, ode_batch_fn = ode_1.inhomogeneous_ornsteinuhlenbeck_batch_fn, 
                                    dt = dt, length = length, num = num, init_range = (50,100),
                                    params = [a, omega, theta, sigma], k = nv_step, N = num_repeat, eqn="inhomogeneous_ornsteinuhlenbeck", dim = '1')
			cond = jnp.concatenate([ts_expand, dW], axis=-1)
			all_ts.append(cond)
			all_ys.append(selected_u)
			all_params.append("{:.4f}_{:.4f}_{:.4f}_{:.4f}".format(a, omega, theta, sigma))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_ICON_SDE_1st_order_tfrecord(name = name, eqn_type = "inhomogeneous_ornsteinuhlenbeck", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys,alpha=nv_step, repeat=num_repeat)

def generate_fluxgate_sensor(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,num_repeat):
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	c_k = jax.random.uniform(next(rng), (eqns,), minval=3, maxval=5)
	lambda_k = jax.random.uniform(next(rng), (eqns,), minval=0.1, maxval=1.0)
	epsilon_k = jax.random.uniform(next(rng), (eqns,), minval = 0.1, maxval = 0.5)
	omega_k = jnp.full_like(epsilon_k, 3.0)
	print('omega_k shape, epsilon_k shape: ', omega_k.shape, epsilon_k.shape)

	all_ts = []; all_ys = []; all_params = []; all_eqn_captions = []
	for i, (c, lambda_, epsilon,omega) in enumerate(zip(c_k, lambda_k, epsilon_k, omega_k)):
		for j in range(quests):
			ts_expand,selected_u, dW = generate_valid_batch(rng, ode_batch_fn = ode_3.fluxgate_sensor_batch_fn, 
									dt = dt, length = length, num = num, init_range = [(-1, 1),(-1,1),(-1,1),(0,0.5), (0,0.5),(0,0.5)],
									params = [c, lambda_, epsilon, omega], k = nv_step, 
									N = num_repeat, eqn = 'fluxgate_sensor', dim = '3')

			cond = jnp.concatenate([ts_expand, dW], axis=-1)

			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))

			all_ts.append(cond)
			all_ys.append(selected_u)
			all_params.append("{:.4f}_{:.4f}_{:.4f}".format(c, lambda_, epsilon))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_ICON_SDE_3rd_order_tfrecord(name = name, eqn_type = "fluxgate_sensor", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
			all_ts = all_ts, all_ys = all_ys,alpha=nv_step, repeat=num_repeat)

def generate_stochastic_lorenz(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,num_repeat):
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	sigma_k = jax.random.uniform(next(rng), (eqns,), minval=5, maxval=15)
	rho_k = jax.random.uniform(next(rng), (eqns,), minval=20, maxval=40)
	beta_k = jax.random.uniform(next(rng), (eqns,), minval = 1.0, maxval = 3.0)
	eta1_k = jax.random.uniform(next(rng), (eqns,), minval=0.1, maxval=2.0)
	eta2_k = jax.random.uniform(next(rng), (eqns,), minval=0.1, maxval=2.0)
	eta3_k = jax.random.uniform(next(rng), (eqns,), minval = 0.1, maxval = 2.0)

	all_ts = []; all_ys = []; all_params = []; all_eqn_captions = []
	for i, (sigma, rho, beta, eta1, eta2, eta3) in enumerate(zip(sigma_k, rho_k, beta_k, eta1_k, eta2_k, eta3_k)):
		for j in range(quests):
			ts_expand, selected_u, dW = generate_valid_batch(rng, ode_batch_fn = ode_3.stochastic_lorenz_batch_fn, 
									dt = dt, length = length, num = num, init_range = [(-1,1),(-1,1),(-1,1)],
									params = [sigma, rho, beta, eta1, eta2, eta3], k = nv_step, 
									N = num_repeat, eqn = 'stochastic_lorenz', dim = '3')

			cond = jnp.concatenate([ts_expand, dW], axis=-1)

			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))

			all_ts.append(cond)
			all_ys.append(selected_u)
			all_params.append("{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}".format(sigma, rho, beta, eta1, eta2, eta3))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_ICON_SDE_3rd_order_tfrecord(name = name, eqn_type = "stochastic_lorenz", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
			all_ts = all_ts, all_ys = all_ys,alpha=nv_step, repeat=num_repeat)

def generate_predator_prey(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,num_repeat):
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	r_k = jax.random.uniform(next(rng), (eqns,), minval=0.3, maxval=0.5)
	a_k = jax.random.uniform(next(rng), (eqns,), minval=0.01, maxval=0.03)
	s_k = jax.random.uniform(next(rng), (eqns,), minval = 0.2, maxval = 0.45)
	b_k = jax.random.uniform(next(rng), (eqns,), minval=0.3, maxval=0.5)
	K_k = jax.random.uniform(next(rng), (eqns,), minval=0.3, maxval=0.5)
	g_k = jax.random.uniform(next(rng), (eqns,), minval = 0.4, maxval = 0.6)
	D_k = jax.random.uniform(next(rng), (eqns,), minval=0.2, maxval=0.5)
	v1_k = jax.random.uniform(next(rng), (eqns,), minval=0.1, maxval=0.5)
	v2_k = jax.random.uniform(next(rng), (eqns,), minval = 0.1, maxval = 0.5)
	sigma1_k = jax.random.uniform(next(rng), (eqns,), minval = 0.01, maxval = 0.5)
	sigma2_k = jax.random.uniform(next(rng), (eqns,), minval = 0.01, maxval = 0.5)
	sigma3_k = jax.random.uniform(next(rng), (eqns,), minval = 0.01, maxval = 0.5)

	all_ts = []; all_ys = []; all_params = []; all_eqn_captions = []
	for i, (r, a, s, b, K, g, D, v1, v2, sigma1, sigma2, sigma3) in enumerate(zip(r_k, a_k, s_k, b_k, K_k, g_k, D_k, v1_k, v2_k, sigma1_k, sigma2_k, sigma3_k)):
		for j in range(quests):
			ts_expand, selected_u, dW = generate_valid_batch(rng, ode_batch_fn = ode_3.predator_prey_batch_fn, 
									dt = dt, length = length, num = num, init_range = [(0, 0.8),(0,0.8),(0,0.8)],
									params = [r, a, s, b, K, g, D, v1, v2, sigma1, sigma2, sigma3], k = nv_step, 
									N = num_repeat, eqn = 'predator_prey', dim = '3')

			cond = jnp.concatenate([ts_expand, dW], axis=-1)

			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))

			all_ts.append(cond)
			all_ys.append(selected_u)
			all_params.append("{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}".format(r, a, s, b, K, g, D, v1, v2, sigma1, sigma2, sigma3))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_ICON_SDE_3rd_order_tfrecord(name = name, eqn_type = 'predator_prey', 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
			all_ts = all_ts, all_ys = all_ys,alpha=nv_step, repeat=num_repeat)


def main(argv):
	# Start timing the entire data generation process
	start_time = time.time()
	print(f"\n{'='*60}")
	print("STARTING DATA GENERATION")
	print(f"{'='*60}\n")
	
	for key, value in FLAGS.__flags.items():
			print(value.name, ": ", value._value, flush=True)
	
	
	name = '{}/{}'.format(FLAGS.dir, FLAGS.name)

	if not os.path.exists(FLAGS.dir):
		os.makedirs(FLAGS.dir)
		
	if 'ornstein_uhlenbeck' in FLAGS.eqn_types:
		print("Generating Ornstein-Uhlenbeck data...")
		eqn_start_time = time.time()
		generate_ornstein_uhlenbeck(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,
							nv_step=FLAGS.nv_step, num_repeat = FLAGS.num_repeat)
		eqn_time = time.time() - eqn_start_time
		print(f"Ornstein-Uhlenbeck generation completed in {eqn_time:.2f} seconds")
	if 'double_well' in FLAGS.eqn_types:
		print("Generating Double Well data...")
		eqn_start_time = time.time()
		generate_double_well(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,
							nv_step=FLAGS.nv_step, num_repeat = FLAGS.num_repeat)
		eqn_time = time.time() - eqn_start_time
		print(f"Double Well generation completed in {eqn_time:.2f} seconds")
	if "coupled_doublewell" in FLAGS.eqn_types:
		generate_coupled_doublewell(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,
							nv_step=FLAGS.nv_step, num_repeat = FLAGS.num_repeat)
	if "mueller_overdamped" in FLAGS.eqn_types:
		generate_mueller_overdamped(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,
							nv_step=FLAGS.nv_step, num_repeat = FLAGS.num_repeat)
	if "duffing_langevin" in FLAGS.eqn_types:
		generate_duffing_langevin(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,
							nv_step=FLAGS.nv_step, num_repeat = FLAGS.num_repeat)
	if "perturbed_nonlinearoscillator" in FLAGS.eqn_types:
		generate_perturbed_nonlinearoscillator(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,
							nv_step=FLAGS.nv_step, num_repeat = FLAGS.num_repeat)
	if "periodic_nonlinearoscillator" in FLAGS.eqn_types:
		generate_periodic_nonlinearoscillator(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,
							nv_step=FLAGS.nv_step, num_repeat = FLAGS.num_repeat)
	if "geombrownian_motion" in FLAGS.eqn_types:
		generate_GMB(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,
							nv_step=FLAGS.nv_step, num_repeat = FLAGS.num_repeat)
	if "inhomogeneous_ornsteinuhlenbeck" in FLAGS.eqn_types:
		generate_inhomogeneous_ornsteinuhlenbeck(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,
							nv_step=FLAGS.nv_step, num_repeat = FLAGS.num_repeat)
	if "fluxgate_sensor" in FLAGS.eqn_types:
		generate_fluxgate_sensor(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,
							nv_step=FLAGS.nv_step, num_repeat = FLAGS.num_repeat)
	if "stochastic_lorenz" in FLAGS.eqn_types:
		print("Generating Stochastic Lorenz data...")
		eqn_start_time = time.time()
		generate_stochastic_lorenz(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,
							nv_step=FLAGS.nv_step, num_repeat = FLAGS.num_repeat)
		eqn_time = time.time() - eqn_start_time
		print(f"Stochastic Lorenz generation completed in {eqn_time:.2f} seconds")
	if "predator_prey" in FLAGS.eqn_types:
		generate_predator_prey(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,
							nv_step=FLAGS.nv_step, num_repeat = FLAGS.num_repeat)
	
	# Calculate and print total data generation time
	total_time = time.time() - start_time
	print(f"\n{'='*60}")
	print("DATA GENERATION COMPLETED")
	print(f"TOTAL DATA GENERATION TIME: {total_time:.4f} seconds")
	print(f"TOTAL DATA GENERATION TIME: {total_time/60:.2f} minutes")
	print(f"TOTAL DATA GENERATION TIME: {total_time/3600:.2f} hours")
	print(f"{'='*60}\n")
		


if __name__ == "__main__":

	import tensorflow as tf
	import os
	os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
	tf.config.set_visible_devices([], device_type='GPU')

	FLAGS = flags.FLAGS
	flags.DEFINE_string('caption_mode', None, 'mode for caption')
	flags.DEFINE_integer('num', 5, 'number of systems in each equation with different initial values') # fixed paramter
	flags.DEFINE_integer('quests', 1, 'number of questions in each operator')
	flags.DEFINE_integer('eqns', 100, 'number of equations')
	flags.DEFINE_integer('length', 40, 'length of trajectory and control')
	flags.DEFINE_integer('nv_step', 5, 'step size of NeurVec')
	flags.DEFINE_float('dt', 0.02, 'time step in dynamics')
	flags.DEFINE_string('name', 'data', 'name of the dataset')
	flags.DEFINE_string('dir', '.', 'name of the directory to save the data')
	flags.DEFINE_list('eqn_types', [], 'list of equations for data generation')
	flags.DEFINE_list('write', [], 'list of features to write')
	flags.DEFINE_integer('num_repeat', 40, 'same initial but different noise')

	flags.DEFINE_integer('seed', 1, 'random seed')

	app.run(main)
