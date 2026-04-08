import numpy as np
import tensorflow as tf
from absl import flags

try:
	FLAGS = flags.FLAGS
	tf_rng_seq = tf.random.Generator.from_seed(FLAGS.seed + 1234)
	print("tf_rng_seq from FLAGS, seed = {}".format(FLAGS.seed + 1234), flush = True)
except:
	tf_rng_seq = tf.random.Generator.from_seed(1234)
	print("tf_rng_seq from default, seed = {}".format(1234), flush = True)

@tf.function
def select_kv(key, val, len_select, select_method):
		'''
		select some k-v pairs from the full set of k-v pairs
		if len_select > len_full, then select all, and pad with 0.
		@ param:
				seed: int
				key: 2D array, [len, kdim]  (len >= len_full)
				val: 2D array, [len, vdim]
				len_full: int
				len_select: int
				select_method: 'random' or 'even' or 'first'
		@ return:
				key_list: the updated list of 2D arrays [len_select, kdim]
				val_list: the updated list of 2D arrays [len_select, vdim]
		'''
		len_full = tf.shape(key)[0]
		if len_select > len_full:
				key = tf.pad(key, [[0, len_select - len_full], [0, 0]])
				val = tf.pad(val, [[0, len_select - len_full], [0, 0]])
		else: # len_select < len_full
			if select_method == 'random':
				seed = tf_rng_seq.make_seeds(1)[:,0]
				index = tf.random.experimental.stateless_shuffle(tf.range(len_full), seed = seed)[0:len_select]
			elif select_method == 'even':
				delta = (len_full - 1) // (len_select - 1)
				index = tf.range(0, len_select) * delta
			elif select_method == 'first':
				index = tf.range(0, len_select)
			key = tf.gather(key, index, axis = 0)
			val = tf.gather(val, index, axis = 0)
		return key, val

# @tf.function
# def select_kv_rollout(key, val, len_select, select_method):
# 	'''
# 	Split the sequence into fixed-length chunks of len_select.
# 	If len_full is not divisible by len_select, the last chunk is padded with zeros.

# 	Args:
# 		key: Tensor of shape [len, kdim]
# 		val: Tensor of shape [len, vdim]
# 	Returns:
# 		key_chunks: Tensor of shape [num_chunks, len_select, kdim]
# 		val_chunks: Tensor of shape [num_chunks, len_select, vdim]
# 	'''
# 	len_full = tf.shape(key)[0]
# 	kdim = tf.shape(key)[1]
# 	vdim = tf.shape(val)[1]

# 	# Compute number of full chunks and padding
# 	num_chunks = tf.math.floordiv(len_full + len_select - 1, len_select)
# 	total_needed = num_chunks * len_select
# 	pad_len = total_needed - len_full

# 	# Pad if needed
# 	key = tf.pad(key, [[0, pad_len], [0, 0]])
# 	val = tf.pad(val, [[0, pad_len], [0, 0]])

# 	# Reshape into chunks
# 	key_chunks = tf.reshape(key, [num_chunks, len_select, kdim])
# 	val_chunks = tf.reshape(val, [num_chunks, len_select, vdim])

# 	return key_chunks, val_chunks

def build_function_kv(demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
											quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
											config, this_config):
	'''
	apply select_kv to all demos and quest
	select cond_len tokens in the range of [0, cond_full_len], 
	if cond_len > cond_full_len, pad with zero
	similarly for qoi
	'''
	demo_cond_k_list = []
	demo_cond_v_list = []
	demo_qoi_k_list = []
	demo_qoi_v_list = []
	for i in range(config['demo_num']):
		this_demo_cond_k, this_demo_cond_v = select_kv(demo_cond_k[i,...], demo_cond_v[i,...], config['demo_cond_len'], this_config['demo_cond_select'])
		this_demo_qoi_k, this_demo_qoi_v = select_kv(demo_qoi_k[i,...], demo_qoi_v[i,...], config['demo_qoi_len'], this_config['demo_qoi_select'])
		demo_cond_k_list.append(this_demo_cond_k)
		demo_cond_v_list.append(this_demo_cond_v)
		demo_qoi_k_list.append(this_demo_qoi_k)
		demo_qoi_v_list.append(this_demo_qoi_v)

	quest_cond_k_list = []
	quest_cond_v_list = []
	quest_qoi_k_list = []
	quest_qoi_v_list = []
	for i in range(config['quest_num']):
		this_quest_cond_k, this_quest_cond_v = select_kv(quest_cond_k[i,...], quest_cond_v[i,...], config['quest_cond_len'], this_config['quest_cond_select'])
		this_quest_qoi_k, this_quest_qoi_v = select_kv(quest_qoi_k[i,...], quest_qoi_v[i,...], config['quest_qoi_len'], this_config['quest_qoi_select'])
		quest_cond_k_list.append(this_quest_cond_k)
		quest_cond_v_list.append(this_quest_cond_v)
		quest_qoi_k_list.append(this_quest_qoi_k)
		quest_qoi_v_list.append(this_quest_qoi_v)
	
	return demo_cond_k_list, demo_cond_v_list, demo_qoi_k_list, demo_qoi_v_list, \
				 quest_cond_k_list, quest_cond_v_list, quest_qoi_k_list, quest_qoi_v_list


# def build_function_kv_rollout(demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
# 											quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
# 											config, this_config):
# 	'''
# 	Applies select_kv in a rollout-compatible way — i.e., slices long sequences
# 	into equal-length chunks and stacks along a new leading dimension.
# 	'''
# 	demo_cond_k_list = []
# 	demo_cond_v_list = []
# 	demo_qoi_k_list = []
# 	demo_qoi_v_list = []

# 	for i in range(config['demo_num']):
# 		this_demo_cond_k, this_demo_cond_v = select_kv(
# 			demo_cond_k[i, ...], demo_cond_v[i, ...],
# 			config['demo_cond_len'], this_config['demo_cond_select']
# 		)
# 		this_demo_qoi_k, this_demo_qoi_v = select_kv(
# 			demo_qoi_k[i, ...], demo_qoi_v[i, ...],
# 			config['demo_qoi_len'], this_config['demo_qoi_select']
# 		)
# 		demo_cond_k_list.append(this_demo_cond_k)
# 		demo_cond_v_list.append(this_demo_cond_v)
# 		demo_qoi_k_list.append(this_demo_qoi_k)
# 		demo_qoi_v_list.append(this_demo_qoi_v)

# 	quest_cond_k_list = []
# 	quest_cond_v_list = []
# 	quest_qoi_k_list = []
# 	quest_qoi_v_list = []

# 	for i in range(config['quest_num']):
# 		this_quest_cond_k, this_quest_cond_v = select_kv(
# 			quest_cond_k[i, ...], quest_cond_v[i, ...],
# 			config['quest_cond_len'], this_config['quest_cond_select']
# 		)
# 		this_quest_qoi_k, this_quest_qoi_v = select_kv(
# 			quest_qoi_k[i, ...], quest_qoi_v[i, ...],
# 			config['quest_qoi_len'], this_config['quest_qoi_select']
# 		)
# 		quest_cond_k_list.append(this_quest_cond_k)
# 		quest_cond_v_list.append(this_quest_cond_v)
# 		quest_qoi_k_list.append(this_quest_qoi_k)
# 		quest_qoi_v_list.append(this_quest_qoi_v)

# 	return demo_cond_k_list, demo_cond_v_list, demo_qoi_k_list, demo_qoi_v_list, \
# 				 quest_cond_k_list, quest_cond_v_list, quest_qoi_k_list, quest_qoi_v_list


def apply_random_demo_num_in_use(config, this_config, demo_cond_mask_list, demo_qoi_mask_list):
	'''
	randomly select the number of demos to be used in the current prompt
	'''
	demo_num_in_use = tf_rng_seq.uniform(shape = (), minval = this_config['demo_num_begin'], maxval = this_config['demo_num_end'], dtype = tf.int32)
	demo_in_use_mask = tf.pad(tf.ones((demo_num_in_use), dtype = tf.int32), [[0, config['demo_num'] - demo_num_in_use]])
	new_demo_cond_mask_list = []
	new_demo_qoi_mask_list = []
	for i in range(config['demo_num']):
		new_demo_cond_mask_list.append(demo_in_use_mask[i] * demo_cond_mask_list[i])
		new_demo_qoi_mask_list.append(demo_in_use_mask[i] * demo_qoi_mask_list[i])
	return new_demo_cond_mask_list, new_demo_qoi_mask_list

def apply_cond_qoi_len_in_use(config, this_config,
															demo_cond_mask_list = None, demo_qoi_mask_list = None, 
															quest_cond_mask_list = None, quest_qoi_mask_list = None,
															demo_cond_len_in_use = None, demo_qoi_len_in_use = None,
															quest_cond_len_in_use = None, quest_qoi_len_in_use = None):
	'''
	apply cond_len_in_use and qoi_len_in_use to the original masks
	'''
	if demo_cond_mask_list is None:
		demo_cond_mask_list = [1 for _ in range(config['demo_num'])]
	if demo_qoi_mask_list is None:
		demo_qoi_mask_list = [1 for _ in range(config['demo_num'])]
	if quest_cond_mask_list is None:
		quest_cond_mask_list = [1 for _ in range(config['quest_num'])]
	if quest_qoi_mask_list is None:
		quest_qoi_mask_list = [1 for _ in range(config['quest_num'])]

	if demo_cond_len_in_use is None:
		demo_cond_len_in_use = tf_rng_seq.uniform(shape = (config['demo_num'],),
															minval = this_config['demo_cond_len_in_use_begin'],
															maxval = this_config['demo_cond_len_in_use_end'], dtype = tf.int32)
	if demo_qoi_len_in_use is None:
		demo_qoi_len_in_use = tf_rng_seq.uniform(shape = (config['demo_num'],),
															minval = this_config['demo_qoi_len_in_use_begin'],
															maxval = this_config['demo_qoi_len_in_use_end'], dtype = tf.int32)
	if quest_cond_len_in_use is None:
		quest_cond_len_in_use = tf_rng_seq.uniform(shape = (config['quest_num'],),
															minval = this_config['quest_cond_len_in_use_begin'],
															maxval = this_config['quest_cond_len_in_use_end'], dtype = tf.int32)
	if quest_qoi_len_in_use is None:
		quest_qoi_len_in_use = tf_rng_seq.uniform(shape = (config['quest_num'],),
															minval = this_config['quest_qoi_len_in_use_begin'],
															maxval = this_config['quest_qoi_len_in_use_end'], dtype = tf.int32)

	new_demo_cond_mask_list = []
	new_demo_qoi_mask_list = []
	for i in range(config['demo_num']):
		demo_cond_mask_i = tf.pad(tf.ones((demo_cond_len_in_use[i],), dtype = tf.int32), [[0, config['demo_cond_len'] - demo_cond_len_in_use[i]]])
		demo_qoi_mask_i = tf.pad(tf.ones((demo_qoi_len_in_use[i],), dtype = tf.int32), [[0, config['demo_qoi_len'] - demo_qoi_len_in_use[i]]])
		new_demo_cond_mask_list.append(demo_cond_mask_i * demo_cond_mask_list[i])
		new_demo_qoi_mask_list.append(demo_qoi_mask_i * demo_qoi_mask_list[i])
	
	new_quest_cond_mask_list = []
	new_quest_qoi_mask_list = []
	for i in range(config['quest_num']):
		quest_cond_mask_i = tf.pad(tf.ones((quest_cond_len_in_use[i],), dtype = tf.int32), [[0, config['quest_cond_len'] - quest_cond_len_in_use[i]]])
		quest_qoi_mask_i = tf.pad(tf.ones((quest_qoi_len_in_use[i],), dtype = tf.int32), [[0, config['quest_qoi_len'] - quest_qoi_len_in_use[i]]])
		new_quest_cond_mask_list.append(quest_cond_mask_i * quest_cond_mask_list[i])
		new_quest_qoi_mask_list.append(quest_qoi_mask_i * quest_qoi_mask_list[i])

	return new_demo_cond_mask_list, new_demo_qoi_mask_list, new_quest_cond_mask_list, new_quest_qoi_mask_list

# def apply_cond_qoi_len_rollout(config, this_config,
# 															demo_cond_mask_list = None, demo_qoi_mask_list = None, 
# 															quest_cond_mask_list = None, quest_qoi_mask_list = None,
# 															demo_cond_len_in_use = None, demo_qoi_len_in_use = None,
# 															quest_cond_len_in_use = None, quest_qoi_len_in_use = None):
# 	'''
# 	Apply cond_len_in_use and qoi_len_in_use to the original masks and stack along first dimension.
# 	If full length is longer than the in-use length, slice and stack.
# 	'''
# 	def split_and_stack(length_total, length_in_use, num_items):
# 		assert length_total % length_in_use == 0, f"{length_total} not divisible by {length_in_use}"
# 		num_chunks = length_total // length_in_use
# 		masks = []
# 		for _ in range(num_chunks):
# 			mask = tf.ones((num_items, length_in_use), dtype=tf.int32)
# 			masks.append(mask)
# 		return masks  # list of tensors shape (B, length_in_use)

# 	if demo_cond_mask_list is None:
# 		demo_cond_mask_list = [1 for _ in range(config['demo_num'])]
# 	if demo_qoi_mask_list is None:
# 		demo_qoi_mask_list = [1 for _ in range(config['demo_num'])]
# 	if quest_cond_mask_list is None:
# 		quest_cond_mask_list = [1 for _ in range(config['quest_num'])]
# 	if quest_qoi_mask_list is None:
# 		quest_qoi_mask_list = [1 for _ in range(config['quest_num'])]

# 	if demo_cond_len_in_use is None:
# 		demo_cond_len_in_use = this_config['demo_cond_len_in_use_end'] - 1
# 	if demo_qoi_len_in_use is None:
# 		demo_qoi_len_in_use = this_config['demo_qoi_len_in_use_end'] - 1
# 	if quest_cond_len_in_use is None:
# 		quest_cond_len_in_use = this_config['quest_cond_len_in_use_end'] - 1
# 	if quest_qoi_len_in_use is None:
# 		quest_qoi_len_in_use = this_config['quest_qoi_len_in_use_end'] - 1

# 	# Stack masks chunk-wise
# 	new_demo_cond_mask_list = split_and_stack(config['demo_cond_len'], demo_cond_len_in_use, config['demo_num'])
# 	new_demo_qoi_mask_list = split_and_stack(config['demo_qoi_len'], demo_qoi_len_in_use, config['demo_num'])
# 	new_quest_cond_mask_list = split_and_stack(config['quest_cond_len'], quest_cond_len_in_use, config['quest_num'])
# 	new_quest_qoi_mask_list = split_and_stack(config['quest_qoi_len'], quest_qoi_len_in_use, config['quest_num'])

# 	return new_demo_cond_mask_list, new_demo_qoi_mask_list, new_quest_cond_mask_list, new_quest_qoi_mask_list



def build_others(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
													quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
													config, this_config):
	demo_cond_k_list, demo_cond_v_list, demo_qoi_k_list, demo_qoi_v_list, \
	quest_cond_k_list, quest_cond_v_list, quest_qoi_k_list, quest_qoi_v_list = \
	build_function_kv(demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, 
										quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
										config, this_config)
	
	demo_cond_mask_list, demo_qoi_mask_list, quest_cond_mask_list, quest_qoi_mask_list = apply_cond_qoi_len_in_use(config, this_config)
	demo_cond_mask_list, demo_qoi_mask_list = apply_random_demo_num_in_use(config, this_config, demo_cond_mask_list, demo_qoi_mask_list)

	return equation, demo_cond_k_list, demo_cond_v_list, demo_qoi_k_list, demo_qoi_v_list, \
									quest_cond_k_list, quest_cond_v_list, quest_qoi_k_list, quest_qoi_v_list, \
									demo_cond_mask_list, demo_qoi_mask_list, quest_cond_mask_list, quest_qoi_mask_list
