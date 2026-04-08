import tensorflow as tf
import numpy as np
import jax.numpy as jnp
from einshape import jax_einshape as einshape
from pprint import pprint
tf.config.set_visible_devices([], device_type='GPU')
import sys
sys.path.append('..')
import utils
from absl import flags

FLAGS = flags.FLAGS

def serialize_element(equation, caption, cond_k, cond_v, qoi_k, qoi_v, count):
	'''
	equation: string describing the equation
	caption: list of strings describing the equation
	cond_k: condition key, 3D, (num, cond_length, cond_k_dim)
	cond_v: condition value, 3D, (num, cond_length, cond_v_dim)
	qoi_k: qoi key, 3D, (num, qoi_length, qoi_k_dim)
	qoi_v: qoi value, 3D, (num, qoi_length, qoi_v_dim)
	'''
	write_list = FLAGS.write
	_print = count < 5

	utils.print_dot(count, freq = 100, marker = "+")
	cond_k = cond_k.astype(np.float32)
	cond_v = cond_v.astype(np.float32)
	qoi_k = qoi_k.astype(np.float32)
	qoi_v = qoi_v.astype(np.float32)

	feature = {
		'equation': tf.train.Feature(bytes_list=tf.train.BytesList(value=[equation.encode("utf-8")])),
		'cond_k': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(cond_k).numpy()])),
		'cond_v': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(cond_v).numpy()])),
		'qoi_k': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(qoi_k).numpy()])),
		'qoi_v': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(qoi_v).numpy()])),
	}

	if _print:
		print('-'*50, count, '-'*50, flush=True)
		print("equation: {}".format(equation), flush=True)
		print("cond_k.shape: {}, cond_v.shape: {}, qoi_k.shape: {}, qoi_v.shape: {}".format(cond_k.shape, cond_v.shape, qoi_k.shape, qoi_v.shape), flush=True)

	example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
	return example_proto.SerializeToString()




def write_SDE_1st_order_tfrecord(name, eqn_type, all_params, all_eqn_captions, all_ts, all_ys, all_errors,alpha,repeat):
	num = all_errors[0].shape[0]

	filename = "{}_{}_{}_{}.tfrecord".format(name, eqn_type, alpha,repeat)
	print("===========" + filename + "===========", flush=True)
	count = 0
	with tf.io.TFRecordWriter(filename) as writer:
		for params, ts_expand, selected_u, error in zip(all_params, all_ts, all_ys, all_errors):
			equation_name = "{}_params_{}_nv_step_{}".format(eqn_type, params,alpha)
			# cond_k_c (100, 49,2) (num, cond_length, cond_k_dim) # before padding, it was 100, 49, 1
			# [0.   0.  ]
			# [0.02 0.  ] ...
			# [0.94 0.  ]
			# [0.96 0.  ]
			# pad the third dimension at the end, with 0.0
			cond_k_c = jnp.pad(ts_expand,((0,0),(0,0),(0,1)),mode = 'constant',constant_values = 0.0)
			cond_k = jnp.pad(cond_k_c,((0,0),(0,0),(0,1)),mode = 'constant',constant_values = 0.0)
			cond_v_c = jnp.pad(selected_u,((0,0),(0,0),(0,1)),mode = 'constant',constant_values = 0.0)
			cond_v = jnp.pad(cond_v_c,((0,0),(0,0),(0,1)),mode = 'constant',constant_values = 0.0)

			error = jnp.pad(error,((0,0),(0,0),(0,1)),mode = 'constant',constant_values = 0.0)
			error = jnp.pad(error,((0,0),(0,0),(0,1)),mode = 'constant',constant_values = 0.0)
			
			
			count += 1
			if np.sum(error) != np.nan:
				s_element= serialize_element(equation = equation_name, caption = None, 
							cond_k = cond_k, cond_v = cond_v, qoi_k = cond_k, qoi_v = error,
							count = count)
				writer.write(s_element)
			else:
				raise Exception("NaN found!")

def write_SDE_2nd_order_tfrecord(name, eqn_type, all_params, all_eqn_captions, all_ts, all_ys, all_errors,alpha, repeat):
	num = all_errors[0].shape[0]

	filename = "{}_{}_{}_{}.tfrecord".format(name, eqn_type, alpha, repeat)
	print("===========" + filename + "===========", flush=True)
	count = 0
	with tf.io.TFRecordWriter(filename) as writer:
		for params, ts_expand, selected_u, error in zip(all_params, all_ts, all_ys, all_errors):
			equation_name = "{}_params_{}_nv_step_{}".format(eqn_type, params,alpha)
			# cond_k_c (100, 49,2) (num, cond_length, cond_k_dim)
			# [0.   0.  ]
			# [0.02 0.  ] ...
			# [0.94 0.  ]
			# [0.96 0.  ]
			# pad the third dimension at the end, with 0.0

			cond_k_c = jnp.pad(ts_expand,((0,0),(0,0),(0,1)),mode = 'constant',constant_values = 0.0)
			cond_k = cond_k_c
			cond_v_c = jnp.pad(selected_u,((0,0),(0,0),(0,1)),mode = 'constant',constant_values = 0.0) 
			cond_v = cond_v_c
			error = jnp.pad(error,((0,0),(0,0),(0,1)),mode = 'constant',constant_values = 0.0)

			count += 1
			if np.sum(error) != np.nan:
				s_element= serialize_element(equation = equation_name, caption = None, 
							cond_k = cond_k, cond_v = cond_v, qoi_k = cond_k, qoi_v = error,
							count = count)
				writer.write(s_element)
			else:
				raise Exception("NaN found!")

def write_SDE_3rd_order_tfrecord(name, eqn_type, all_params, all_eqn_captions, all_ts, all_ys, all_errors,alpha, repeat):
	num = all_errors[0].shape[0]

	filename = "{}_{}_{}_{}.tfrecord".format(name, eqn_type, alpha, repeat)
	print("===========" + filename + "===========", flush=True)
	count = 0
	with tf.io.TFRecordWriter(filename) as writer:
		for params, ts_expand, selected_u, error in zip(all_params, all_ts, all_ys, all_errors):
			equation_name = "{}_params_{}_nv_step_{}".format(eqn_type, params,alpha)
			
			cond_k_c = ts_expand
			cond_k = cond_k_c
			cond_v_c = selected_u
			cond_v_c = selected_u
			cond_v = cond_v_c
			
			# print('error',error.shape)

			count += 1
			if np.sum(error) != np.nan:
				s_element= serialize_element(equation = equation_name, caption = None, 
							cond_k = cond_k, cond_v = cond_v, qoi_k = ts_expand, qoi_v = error,
							count = count)
				writer.write(s_element)
			else:
				raise Exception("NaN found!")