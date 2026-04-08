import torch
import tensorflow as tf
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
tf.config.set_visible_devices([], device_type='GPU')
from pprint import pprint

import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np
from functools import partial
import haiku as hk
import optax
import pytz
from datetime import datetime
import pickle
from absl import app, flags, logging
from collections import namedtuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from einshape import jax_einshape as einshape
import gc
import glob
from pprint import pprint
import os
import glob

import sys
sys.path.append('../')
import utils
import plot

newData = namedtuple('Data', ['demo_cond_k', 'demo_cond_v', 'demo_cond_mask', 
						'demo_qoi_k', 'demo_qoi_v', 'demo_qoi_mask',
						'quest_cond_k', 'quest_cond_v', 'quest_cond_mask',
						'quest_qoi_k', 'quest_qoi_mask',])

def append_dict_list(dict, key, value):
	if key not in dict:
		dict[key] = []
	dict[key].append(value)
	return dict

def get_key(eqn_name):
	'''
	get the key for the eqn_name
	@param 
		eqn_name: string, the name of the equation
	@return key: tuple
	'''
	if isinstance(eqn_name, np.ndarray):
		pass
	elif isinstance(eqn_name, str):
		pass
	else:
		eqn_name = eqn_name.numpy().decode('utf-8')

	eqn_name_split = eqn_name.split("_")
	eqn_name_clean = "_".join(eqn_name_split[:2])
	key = (eqn_name_clean,)
	
	return key


def write_into_dict(result_dict, runner, equation, all_caption, all_data, label, test_demo_num_list, test_caption_id_list, split_data):
	if FLAGS.backend == 'jax':
		# flatten the batch dimension
		all_data_flat = tree.tree_map(lambda x: einshape('ij...->(ij)...', x), all_data)
		label_flat = einshape('ij...->(ij)...', label)
	else:
		all_data_flat = all_data
		label_flat = label
	
	test_caption_id_list_in_use = test_caption_id_list.copy()
	# write ground truth and mask into dict
	print('total number i: ', label_flat.shape[0], equation.shape, equation[0])
	for i in range(label_flat.shape[0]):
		eqn_key = get_key(equation[i])
		append_dict_list(result_dict, (*eqn_key, 'ground_truth'), label_flat[i,0])


	if 'quest' in FLAGS.write:
		for i in range(label_flat.shape[0]):
			eqn_key = get_key(equation[i])
			# append_dict_list(result_dict, (*eqn_key, 'cond_k'), all_data_flat.quest_cond_k[i,0])
			append_dict_list(result_dict, (*eqn_key, 'cond_v'), all_data_flat.quest_cond_v[i,0])
			# append_dict_list(result_dict, (*eqn_key, 'cond_mask'), all_data_flat.quest_cond_mask[i,0])
			# append_dict_list(result_dict, (*eqn_key, 'qoi_k'), all_data_flat.quest_qoi_k[i,0])

	if 'demo' in FLAGS.write:
		for i in range(label_flat.shape[0]):
			eqn_key = get_key(equation[i])
			append_dict_list(result_dict, (*eqn_key, 'demo_cond_k'), all_data_flat.demo_cond_k[i])
			append_dict_list(result_dict, (*eqn_key, 'demo_cond_v'), all_data_flat.demo_cond_v[i])
			append_dict_list(result_dict, (*eqn_key, 'demo_cond_mask'), all_data_flat.demo_cond_mask[i])
			append_dict_list(result_dict, (*eqn_key, 'demo_qoi_k'), all_data_flat.demo_qoi_k[i])
			append_dict_list(result_dict, (*eqn_key, 'demo_qoi_v'), all_data_flat.demo_qoi_v[i])
			append_dict_list(result_dict, (*eqn_key, 'demo_qoi_mask'), all_data_flat.demo_qoi_mask[i])

	if 'equation' in FLAGS.write:
		for i in range(label_flat.shape[0]):
			eqn_key = get_key(equation[i])
			append_dict_list(result_dict, (*eqn_key, 'equation'), equation[i])

	# write error into dict, test_caption_id_list_in_use = [-1] or [0] or [-1,0]
	
	# -1 indicates no caption
	if -1 in test_caption_id_list_in_use:
		test_caption_id_list_in_use.remove(-1) # remove -1 from the list
		for demo_num, caption_id, caption, data in split_data(all_caption, all_data, test_demo_num_list, [0]):    
			this_error, this_pred = runner.get_error(data, label, with_caption = False, return_pred = True)
			if FLAGS.backend == 'jax':
				this_error = einshape('ij...->(ij)...', this_error)
				this_pred = einshape('ij...->(ij)...', this_pred)
			for i in range(this_error.shape[0]):
				eqn_key = get_key(equation[i])
				append_dict_list(result_dict, (*eqn_key, 'error', demo_num, -1), this_error[i]) # -1 means no caption
				append_dict_list(result_dict, (*eqn_key, 'pred', demo_num, -1), this_pred[i])

	# count len(test_caption_id_list_in_use other than -1), usually 0 or 1
	for demo_num, caption_id, caption, data in split_data(all_caption, all_data, test_demo_num_list, test_caption_id_list_in_use):    
		this_error, this_pred = runner.get_error(data, label, with_caption = True, return_pred = True)
		if FLAGS.backend == 'jax':
			this_error = einshape('ij...->(ij)...', this_error)
			this_pred = einshape('ij...->(ij)...', this_pred)
		for i in range(this_error.shape[0]):
			eqn_key = get_key(equation[i])
			append_dict_list(result_dict, (*eqn_key, 'error', demo_num, caption_id), this_error[i])
			append_dict_list(result_dict, (*eqn_key, 'pred', demo_num, caption_id), this_pred[i])
	

def _slice_data_to_demo_num(data, demo_num):
	"""Slice data to use only the first demo_num demos (for sweep_demo_nums in rollout)."""
	if not hasattr(data, 'demo_cond_k') or data.demo_cond_k is None:
		return data
	# Rollout data layout: (num_devices, batch_per_device, num_demos, ...); demo axis is 2.
	# Non-rollout layout: (batch, num_demos, ...); demo axis is 1.
	shp = data.demo_cond_k.shape
	if len(shp) >= 3:
		num_d = shp[2]
		demo_axis = 2
	else:
		num_d = shp[1]
		demo_axis = 1
	if demo_num >= num_d:
		return data
	kwargs = {}
	for name in data._fields:
		v = getattr(data, name)
		if name.startswith('demo_') and hasattr(v, 'shape') and len(v.shape) > demo_axis:
			# Slice demo axis only so batch/devices dimensions stay unchanged
			kwargs[name] = np.take(v, np.arange(demo_num), axis=demo_axis)
		else:
			kwargs[name] = v
	return type(data)(**kwargs)


def write_rollout_into_dict(result_dict, all_equations, concat_pred, concat_label, concat_data, test_demo_num_list, pred_by_demo=None):
	"""
	Populate result_dict with the concatenated rollout results (aligned coarse + rollout pred).
	If pred_by_demo is not None (sweep_demo_nums), it must be a dict demo_num -> concat_pred array;
	then predictions are written for each demo_num. Otherwise concat_pred is used for a single demo count.
	"""
	n_samples = concat_label.shape[0]
	for i in range(n_samples):
		eqn_key = get_key(all_equations[i])
		append_dict_list(result_dict, (*eqn_key, 'ground_truth'), concat_label[i, 0])
	if 'quest' in FLAGS.write:
		for i in range(n_samples):
			eqn_key = get_key(all_equations[i])
			append_dict_list(result_dict, (*eqn_key, 'cond_v'), concat_data.quest_cond_v[i, 0])
	if 'equation' in FLAGS.write:
		for i in range(n_samples):
			eqn_key = get_key(all_equations[i])
			append_dict_list(result_dict, (*eqn_key, 'equation'), all_equations[i])
	if pred_by_demo is not None:
		for demo_num, concat_pred_d in pred_by_demo.items():
			for i in range(n_samples):
				eqn_key = get_key(all_equations[i])
				append_dict_list(result_dict, (*eqn_key, 'pred', demo_num, -1), concat_pred_d[i, 0] if concat_pred_d.ndim >= 2 else concat_pred_d[i])
	else:
		num_demos_in_data = concat_data.demo_cond_k.shape[1] if hasattr(concat_data, 'demo_cond_k') and concat_data.demo_cond_k is not None else (int(test_demo_num_list[0]) if test_demo_num_list else 1)
		demo_num = num_demos_in_data
		for i in range(n_samples):
			eqn_key = get_key(all_equations[i])
			append_dict_list(result_dict, (*eqn_key, 'pred', demo_num, -1), concat_pred[i, 0])


def run_analysis():
	# Track inference time for this function only
	total_inference_time = 0.0

	utils.set_seed(FLAGS.seed)
	if FLAGS.correction:
		from dataloader_fmint_SDE import DataProvider, print_eqn_caption, split_data # import in function to enable flags in dataloader
	else:
		from dataloader_fmint_SDE import DataProvider, print_eqn_caption, split_data # import in function to enable flags in dataloader
	

	test_data_dirs = FLAGS.test_data_dirs
	test_file_names = ["{}/{}".format(i, j) for i in test_data_dirs for j in FLAGS.test_data_globs]

	print("test_file_names: ", flush=True)
	pprint(test_file_names)

	test_config = utils.load_json("../config_data/" + FLAGS.test_config_filename)
	model_config = utils.load_json("../config_model/" + FLAGS.model_config_filename)

	if 'cap' not in FLAGS.loss_mode:
		model_config['caption_len'] = 0
		test_config['load_list'] = []


	print('==============data config==============', flush = True)
	print("test_config: ", flush=True)
	pprint(test_config)
	print('==============data config end==============', flush = True)

	print('-----------------------model config-----------------------')
	print("model_config: ", flush=True)
	pprint(model_config)
	print('-----------------------model config end-----------------------')


	if FLAGS.backend == 'jax':
		optimizer = optax.adamw(0.0001) # dummy optimizer
		import jax
		data_num_devices = len(jax.devices())
	elif FLAGS.backend == 'torch':
		# dummy optimizer
		opt_config = {'peak_lr': 0.001,
									'end_lr': 0,
									'warmup_steps': 10,
									'decay_steps': 100,
									'gnorm_clip': 1,
									'weight_decay': 0.0001,
									}
		data_num_devices = 0
	else:
		raise ValueError("backend {} not supported".format(FLAGS.backend))

	test_demo_num_list = [int(i) for i in FLAGS.test_demo_num_list]
	test_caption_id_list = [int(i) for i in FLAGS.test_caption_id_list]

	test_data = DataProvider(seed = FLAGS.seed + 10,
														config = test_config,
														file_names = test_file_names,
														batch_size = FLAGS.batch_size,
														deterministic = True,
														drop_remainder = False, 
														shuffle_dataset = False,
														num_epochs=1,
														shuffle_buffer_size=10,
														num_devices=data_num_devices,
														real_time = True,
														caption_home_dir = '../data_preparation',
														name = 'analysis'
													)
	
	equation, caption, data, label = test_data.get_next_data()
	print('equation in analysis:', equation)
	print_eqn_caption(equation, caption, decode = False)
	print(tree.tree_map(lambda x: x.shape, data)) 

	if FLAGS.model in ['icon', 'icon_scale', 'icon_scale_surrogate']:
		from runner_jax import Runner_vanilla
		runner = Runner_vanilla(seed = FLAGS.seed,
										model = FLAGS.model,
										data = data,
										model_config = model_config,
										optimizer = optimizer,
										trainable_mode = 'all',
										)
	elif FLAGS.model in ['icon_lm']:
		from runner_jax import Runner_lm
		runner = Runner_lm(seed = FLAGS.seed,
										model = FLAGS.model,
										data = data,
										model_config = model_config,
										optimizer = optimizer,
										trainable_mode = 'all',
										loss_mode = FLAGS.loss_mode,
										)
	elif FLAGS.model in ['gpt2']:
		from runner_torch import Runner
		runner = Runner(data, model_config, opt_config = opt_config, 
										model_name = FLAGS.model, pretrained = True, 
										trainable_mode = 'all',
										loss_mode = FLAGS.loss_mode,
										)
	else:
		raise ValueError("model {} not supported".format(FLAGS.model))
	

	runner.restore(FLAGS.restore_dir, FLAGS.restore_step, restore_opt_state=False)
	result_dict = {}
	
	test_demo_num_list = [int(i) for i in FLAGS.test_demo_num_list]
	if not FLAGS.sweep_demo_nums:
		test_demo_num_list = [4]
	test_caption_id_list = [int(i) for i in FLAGS.test_caption_id_list]

	write_into_dict(result_dict, runner, equation, caption, data, label, 
									test_demo_num_list, test_caption_id_list, split_data)
	
	if not os.path.exists(FLAGS.analysis_dir):
		os.makedirs(FLAGS.analysis_dir)

# 加一个对error correct后的plot
	# Time the inference
	start_time = time.time()
	pred = runner.get_pred(data, with_caption=False) # (num_devices, batch_on_each_device, ...)
	inference_time = time.time() - start_time
	total_inference_time += inference_time
	print(f"Analysis inference time for this batch: {inference_time:.4f} seconds")
	if FLAGS.backend == 'jax': # additional dimension for num_devices
		this_data = tree.tree_map(lambda x: einshape('ij...->(ij)...', np.array(x)), data)
		this_label = einshape('ij...->(ij)...', np.array(label))
		this_pred = einshape('ij...->(ij)...', np.array(pred))
	else: # no additional dimension for num_devices
		this_data = tree.tree_map(lambda x: np.array(x), data)
		this_label = np.array(label)
		this_pred = np.array(pred)
	for fij in range(10):
		this_equation_ij = equation[fij] if type(equation[fij]) == str else equation[fij].numpy().decode('utf-8')
		this_caption_ij = caption[fij] if type(caption[fij]) == str else caption[fij].numpy().decode('utf-8')
		this_data_ij = tree.tree_map(lambda x: x[fij], this_data)
		# Plotting only when FLAGS.correction is True; skipped otherwise
		if FLAGS.correction:
			figure_correct = plot.correction_plot_data(this_equation_ij, this_caption_ij,
															this_data_ij, this_label[fij], this_pred[fij], test_config, to_tfboard = False)
			figure_sde = plot.SDE_plot_pathwise(this_equation_ij, this_caption_ij,
														this_data_ij, this_label[fij], this_pred[fij], test_config, to_tfboard = False)
		
			figure_correct.savefig(FLAGS.analysis_dir+"/error_{}_{}.pdf".format(this_equation_ij,fij))
			figure_sde.savefig(FLAGS.analysis_dir+"/sde_{}_{}.pdf".format(this_equation_ij, fij))
			# pattern = os.path.join(FLAGS.analysis_dir, f"error_{this_equation_ij}_*.pdf")
			# existing_files = glob.glob(pattern)

			# # Get index based on current count
			# ij = len(existing_files)

			# # Now save the figures
			# figure_correct.savefig(os.path.join(FLAGS.analysis_dir, f"error_{this_equation_ij}_{ij}.pdf"))
			# figure_sde.savefig(os.path.join(FLAGS.analysis_dir, f"sde_{this_equation_ij}_{ij}.pdf"))
		else:
			pass
			# figure_sde = plot.icon_ODE_plot_data(this_equation_ij, this_caption_ij,
			# 											this_data_ij, this_label[fij], this_pred[fij], test_config, to_tfboard = False)
			# figure_sde.savefig(FLAGS.analysis_dir+"/sde_{}_{}.pdf".format(this_equation_ij, fij))
		

	read_step = 0
	while True:
		utils.print_dot(read_step)
		read_step += 1
		try:
			equation, caption, data, label = test_data.get_next_data()
		except StopIteration:
			break
		write_into_dict(result_dict, runner, equation, caption, data, label, 
										test_demo_num_list, test_caption_id_list, split_data)

	for key, value in result_dict.items():
		result_dict[key] = np.array(value)
	
	# Print total inference time for analysis function
	print(f"\n{'='*60}")
	print(f"ANALYSIS TOTAL INFERENCE TIME: {total_inference_time:.4f} seconds")
	print(f"ANALYSIS TOTAL INFERENCE TIME: {total_inference_time/60:.2f} minutes")
	print(f"{'='*60}\n")
	
	return result_dict


def run_convergence():

    utils.set_seed(FLAGS.seed)
    if FLAGS.correction:
        from dataloader_fmint_SDE import DataProvider, print_eqn_caption, split_data
    else:
        from dataloader_fmint_SDE import DataProvider, print_eqn_caption, split_data

    test_data_dirs = FLAGS.test_data_dirs
    test_file_names = ["{}/{}".format(i, j) for i in test_data_dirs for j in FLAGS.test_data_globs]

    print("test_file_names: ", flush=True)
    pprint(test_file_names)

    test_config = utils.load_json("../config_data/" + FLAGS.test_config_filename)
    model_config = utils.load_json("../config_model/" + FLAGS.model_config_filename)

    if 'cap' not in FLAGS.loss_mode:
        model_config['caption_len'] = 0
        test_config['load_list'] = []

    print('==============data config==============', flush=True)
    print("test_config: ", flush=True)
    pprint(test_config)
    print('==============data config end==============', flush=True)

    print('-----------------------model config-----------------------')
    print("model_config: ", flush=True)
    pprint(model_config)
    print('-----------------------model config end-----------------------')

    if FLAGS.backend == 'jax':
        optimizer = optax.adamw(0.0001)
        import jax
        data_num_devices = len(jax.devices())
    elif FLAGS.backend == 'torch':
        opt_config = {'peak_lr': 0.001,
                      'end_lr': 0,
                      'warmup_steps': 10,
                      'decay_steps': 100,
                      'gnorm_clip': 1,
                      'weight_decay': 0.0001,}
        data_num_devices = 0
    else:
        raise ValueError("backend {} not supported".format(FLAGS.backend))

    test_demo_num_list = [int(i) for i in FLAGS.test_demo_num_list]
    if not FLAGS.sweep_demo_nums:
        test_demo_num_list = [4]
    test_caption_id_list = [int(i) for i in FLAGS.test_caption_id_list]

    test_data = DataProvider(
        seed=FLAGS.seed + 10,
        config=test_config,
        file_names=test_file_names,
        batch_size=FLAGS.batch_size,
        deterministic=True,
        drop_remainder=False,
        shuffle_dataset=False,
        num_epochs=1,
        shuffle_buffer_size=10,
        num_devices=data_num_devices,
        real_time=True,
        caption_home_dir='../data_preparation',
		name = 'analysis'
    )

    # get_test_iterator returns a generator
    iterator = test_data.get_test_iterator()
    equation, caption, data, label = next(iterator)

    equation = equation[0]
    caption = caption[0]
    # print('caption and equation in convergence: ',caption, equation)
    # print(tree.tree_map(lambda x: x.shape, data))

    if FLAGS.model in ['icon', 'icon_scale', 'icon_scale_surrogate']:
        from runner_jax import Runner_vanilla
        runner = Runner_vanilla(
            seed=FLAGS.seed,
            model=FLAGS.model,
            data=data,
            model_config=model_config,
            optimizer=optimizer,
            trainable_mode='all',
        )
    elif FLAGS.model in ['icon_lm']:
        from runner_jax import Runner_lm
        runner = Runner_lm(
            seed=FLAGS.seed,
            model=FLAGS.model,
            data=data,
            model_config=model_config,
            optimizer=optimizer,
            trainable_mode='all',
            loss_mode=FLAGS.loss_mode,
        )
    elif FLAGS.model in ['gpt2']:
        from runner_torch import Runner
        runner = Runner(
            data,
            model_config,
            opt_config=opt_config,
            model_name=FLAGS.model,
            pretrained=True,
            trainable_mode='all',
            loss_mode=FLAGS.loss_mode,
        )
    else:
        raise ValueError("model {} not supported".format(FLAGS.model))

    runner.restore(FLAGS.restore_dir, FLAGS.restore_step, restore_opt_state=False)
    result_dict = {}

    write_into_dict(result_dict, runner, equation, caption, data, label,
                    test_demo_num_list, test_caption_id_list, split_data)

    if not os.path.exists(FLAGS.analysis_dir):
        os.makedirs(FLAGS.analysis_dir)

# 加一个对error correct后的plot
    pred = runner.get_pred(data, with_caption=False) # (num_devices, batch_on_each_device, ...)
    if FLAGS.backend == 'jax': # additional dimension for num_devices
        this_data = tree.tree_map(lambda x: einshape('ij...->(ij)...', np.array(x)), data)
        this_label = einshape('ij...->(ij)...', np.array(label))
        this_pred = einshape('ij...->(ij)...', np.array(pred))
    else: # no additional dimension for num_devices
        this_data = tree.tree_map(lambda x: np.array(x), data)
        this_label = np.array(label)
        this_pred = np.array(pred)
    
    this_pred = np.expand_dims(this_pred, axis=1)
    for fij in range(2):
        batch_indices = slice(FLAGS.num_repeat*fij, FLAGS.num_repeat*fij + FLAGS.num_repeat)

        # Handle batched strings or tensors for equation
        these_equations = [
            eq if isinstance(eq, str) else eq.numpy().decode('utf-8')
            for eq in equation[batch_indices]
        ]

        # Same for captions
        these_captions = [
            cap if isinstance(cap, str) else cap.numpy().decode('utf-8')
            for cap in caption[batch_indices]
        ]
        # Apply tree_map across batched data
        these_data = tree.tree_map(lambda x: x[batch_indices], this_data)

        # Slice labels and predictions as well
        these_labels = this_label[batch_indices]
        these_preds = this_pred[batch_indices]

        # Plotting only when FLAGS.correction is True; skipped otherwise
        if FLAGS.correction:
            figure_sde_expectation = plot.SDE_plot_expectation(these_equations[0], these_captions,
                                                        these_data, these_labels, these_preds, test_config, FLAGS.num_repeat, to_tfboard = False)
            
            figure_sde_expectation.savefig(FLAGS.analysis_dir+"/Allinone_SDE_{}_{}.pdf".format(these_equations[0],fij))
        # else:
        #     figure_ode = plot.icon_ODE_plot_data(this_equation_ij, this_caption_ij,
        #                                         this_data_ij, this_label[fij], this_pred[fij], test_config, to_tfboard = False)
        #     figure_ode.savefig(FLAGS.analysis_dir+"/sde_{}_{}.pdf".format(this_equation_ij, fij))
       
    read_step = 0
    for equation, caption, data, label in test_data.get_test_iterator():
        utils.print_dot(read_step)
        read_step += 1
        # print('equation in convergence:', equation)
        write_into_dict(result_dict, runner, equation[0], caption, data, label,
                        test_demo_num_list, test_caption_id_list, split_data)

    for key, value in result_dict.items():
        result_dict[key] = np.array(value)
    return result_dict

def run_rollout():
    # Track inference time for this function only
    total_inference_time = 0.0

    utils.set_seed(FLAGS.seed)
    if FLAGS.correction:
        from dataloader_fmint_SDE import DataProvider, print_eqn_caption, split_data
    else:
        from dataloader_fmint_SDE import DataProvider, print_eqn_caption, split_data

    test_data_dirs = FLAGS.test_data_dirs
    test_file_names = ["{}/{}".format(i, j) for i in test_data_dirs for j in FLAGS.test_data_globs]

    print("test_file_names: ", flush=True)
    pprint(test_file_names)

    test_config = utils.load_json("../config_data/" + FLAGS.test_config_filename)
    model_config = utils.load_json("../config_model/" + FLAGS.model_config_filename)

    if 'cap' not in FLAGS.loss_mode:
        model_config['caption_len'] = 0
        test_config['load_list'] = []

    print('==============data config==============', flush=True)
    print("test_config: ", flush=True)
    pprint(test_config)
    print('==============data config end==============', flush=True)

    print('-----------------------model config-----------------------')
    print("model_config: ", flush=True)
    pprint(model_config)
    print('-----------------------model config end-----------------------')

    if FLAGS.backend == 'jax':
        optimizer = optax.adamw(0.0001)
        import jax
        data_num_devices = len(jax.devices())
    elif FLAGS.backend == 'torch':
        opt_config = {'peak_lr': 0.001,
                      'end_lr': 0,
                      'warmup_steps': 10,
                      'decay_steps': 100,
                      'gnorm_clip': 1,
                      'weight_decay': 0.0001,}
        data_num_devices = 0
    else:
        raise ValueError("backend {} not supported".format(FLAGS.backend))

    test_demo_num_list = [int(i) for i in FLAGS.test_demo_num_list]
    if not FLAGS.sweep_demo_nums:
        test_demo_num_list = [4]
    test_caption_id_list = [int(i) for i in FLAGS.test_caption_id_list]

    test_data = DataProvider(
        seed=FLAGS.seed + 10,
        config=test_config,
        file_names=test_file_names,
        batch_size=FLAGS.batch_size,
        deterministic=True,
        drop_remainder=False,
        shuffle_dataset=False,
        num_epochs=1,
        shuffle_buffer_size=10,
        num_devices=data_num_devices,
        real_time=True,
        caption_home_dir='../data_preparation',
		name = 'analysis'
    )

    # Run rollout over all batches from the dataset (full dataset length)
    iterator = test_data.get_rollout_iterator()
    result_dict = {}
    if not os.path.exists(FLAGS.analysis_dir):
        os.makedirs(FLAGS.analysis_dir)
    first_batch_plot_data = None  # for saving rollout figures from first batch only
    total_batches_processed = 0

    for batch_idx, (equation, caption, data, label, num_realization) in enumerate(iterator):
        # collapse device dimension so equation/caption are (n,) for this batch
        equation = equation[0] if hasattr(equation, 'ndim') and equation.ndim > 1 else equation
        caption = caption[0] if hasattr(caption, 'ndim') and caption.ndim > 1 else caption

        if batch_idx == 0:
            if FLAGS.model in ['icon', 'icon_scale', 'icon_scale_surrogate']:
                from runner_jax import Runner_vanilla
                runner = Runner_vanilla(
                    seed=FLAGS.seed,
                    model=FLAGS.model,
                    data=data,
                    model_config=model_config,
                    optimizer=optimizer,
                    trainable_mode='all',
                )
            elif FLAGS.model in ['icon_lm']:
                from runner_jax import Runner_lm
                runner = Runner_lm(
                    seed=FLAGS.seed,
                    model=FLAGS.model,
                    data=data,
                    model_config=model_config,
                    optimizer=optimizer,
                    trainable_mode='all',
                    loss_mode=FLAGS.loss_mode,
                )
            elif FLAGS.model in ['gpt2']:
                from runner_torch import Runner
                runner = Runner(
                    data,
                    model_config,
                    opt_config=opt_config,
                    model_name=FLAGS.model,
                    pretrained=True,
                    trainable_mode='all',
                    loss_mode=FLAGS.loss_mode,
                )
            else:
                raise ValueError("model {} not supported".format(FLAGS.model))
            runner.restore(FLAGS.restore_dir, FLAGS.restore_step, restore_opt_state=False)

        all_preds = []
        all_labels = []
        all_data = []
        all_equations = []
        all_captions = []
        sweep_rollout = FLAGS.sweep_demo_nums and len(test_demo_num_list) > 1
        all_preds_by_demo = {d: [] for d in test_demo_num_list} if sweep_rollout else None

        division_size = num_realization
        tot_split = equation.shape[0] // division_size
        print(f'rollout batch {batch_idx}: total split: {tot_split}, num_realization: {num_realization}, label.shape: {label.shape}' + (' (sweep_demo_nums)' if sweep_rollout else ''))
        if batch_idx == 0:
            print('this data before processing: ', tree.tree_map(lambda x: x.shape, data))

        for split_idx in range(tot_split):
            start_idx = split_idx * division_size
            end_idx = (split_idx + 1) * division_size
            batch_indices = slice(start_idx, end_idx)

            this_data = tree.tree_map(lambda x: x[:,batch_indices], data)
            this_label = label[:,batch_indices]

            if test_config['rollout_update'] and split_idx > 0:
                prev_start_idx = (split_idx - 1) * division_size
                prev_end_idx = split_idx * division_size
                prev_batch_indices = slice(prev_start_idx, prev_end_idx)
                prev_data = tree.tree_map(lambda x: x[:,prev_batch_indices], data)
                prev_label = label[:,prev_batch_indices]

                if batch_idx == 0 and split_idx == 1:
                    print('this data: ', this_data.demo_cond_k.shape)
                    print('prev data: ', prev_data.demo_cond_k.shape)

                this_data.demo_cond_k[:,:,:,:,0] += prev_data.demo_cond_k[:, :, :, - 1, 0][:, :, :, np.newaxis]
                this_data.demo_cond_v[:,:,:,:,:] -= prev_data.demo_qoi_v[:, :,:,-1,:][:,:,:, np.newaxis]
                this_data.demo_qoi_k[:,:,:,:,0] += prev_data.demo_qoi_k[:, :, :, - 1, 0][:, :, :, np.newaxis]
                this_data.demo_qoi_v[:,:,:,:,:] += prev_data.demo_qoi_v[:, :, :, - 1, :][:,:,:, np.newaxis]
                this_data.quest_cond_k[:,:,:,:,0] += prev_data.quest_cond_k[:, :, :, - 1, 0][:, :,:,  np.newaxis]
                this_data.quest_cond_v[:,:,:,:,:] -= prev_label[:, :, :, - 1, :][:,:,:, np.newaxis]
                this_data.quest_qoi_k[:,:,:,:,0] += prev_data.quest_qoi_k[:, :, :, - 1, 0][:, :,:,  np.newaxis]
                this_label += prev_label[:, :, :, - 1, :][:,:,:, np.newaxis]

            these_equations = [
                eq if isinstance(eq, str) else (eq.numpy().decode('utf-8') if hasattr(eq, 'numpy') else str(eq))
                for eq in equation[batch_indices]
            ]
            these_captions = [
                cap if isinstance(cap, str) else (cap.numpy().decode('utf-8') if hasattr(cap, 'numpy') else str(cap))
                for cap in caption[batch_indices]
            ]
            num_demos = this_data.demo_cond_k.shape[1] if hasattr(this_data, 'demo_cond_k') and this_data.demo_cond_k is not None else 4
            demo_nums_to_run = [d for d in test_demo_num_list if d <= num_demos] if sweep_rollout else []

            if sweep_rollout and len(demo_nums_to_run) > 0:
                for demo_num in demo_nums_to_run:
                    this_data_demo = _slice_data_to_demo_num(this_data, demo_num)
                    start_time = time.time()
                    pred = runner.get_pred(this_data_demo, with_caption=False)
                    inference_time = time.time() - start_time
                    total_inference_time += inference_time
                    if FLAGS.backend == 'jax':
                        this_pred = einshape('ij...->(ij)...', np.array(pred))
                    else:
                        this_pred = np.array(pred)
                    this_pred = np.expand_dims(this_pred, axis=1)
                    all_preds_by_demo[demo_num].append(this_pred)
                if split_idx == 0 and batch_idx == 0:
                    print(f"Rollout inference time for this segment (sweep {len(demo_nums_to_run)} demo nums): {inference_time * len(demo_nums_to_run):.4f} seconds")
                # build filtered_data and all_labels once per segment (same for all demo counts)
                if FLAGS.backend == 'jax':
                    this_data_np = tree.tree_map(lambda x: einshape('ij...->(ij)...', np.array(x)), this_data)
                    this_label_np = einshape('ij...->(ij)...', np.array(this_label))
                else:
                    this_data_np = tree.tree_map(lambda x: np.array(x), this_data)
                    this_label_np = np.array(this_label)
                filtered_data = newData(**{k: getattr(this_data_np, k) for k in newData._fields})
                all_data.append(filtered_data)
                all_labels.append(np.expand_dims(this_label_np, axis=1) if this_label_np.ndim == 2 else this_label_np)
            else:
                start_time = time.time()
                pred = runner.get_pred(this_data, with_caption=False)
                inference_time = time.time() - start_time
                total_inference_time += inference_time
                print(f"Rollout inference time for this segment: {inference_time:.4f} seconds")

                if FLAGS.backend == 'jax':
                    this_data_np = tree.tree_map(lambda x: einshape('ij...->(ij)...', np.array(x)), this_data)
                    this_label = einshape('ij...->(ij)...', np.array(this_label))
                    this_pred = einshape('ij...->(ij)...', np.array(pred))
                else:
                    this_data_np = tree.tree_map(lambda x: np.array(x), this_data)
                    this_label = np.array(this_label)
                    this_pred = np.array(pred)
                this_pred = np.expand_dims(this_pred, axis=1)
                this_data = this_data_np
                expected_fields = newData._fields
                filtered_data = newData(**{k: getattr(this_data, k) for k in expected_fields})
                all_preds.append(this_pred)
                all_labels.append(this_label)
                all_data.append(filtered_data)

            all_equations += these_equations
            all_captions += these_captions

        if sweep_rollout and all_preds_by_demo and any(all_preds_by_demo[d] for d in test_demo_num_list):
            concat_label = np.concatenate(all_labels, axis=2)
            pred_by_demo = {}
            for d in test_demo_num_list:
                if all_preds_by_demo[d]:
                    pred_by_demo[d] = np.concatenate(all_preds_by_demo[d], axis=2)
            concat_data = tree.tree_map(lambda *xs: np.concatenate(xs, axis=2), *all_data)
            # use max demo count for first-batch plot; fallback to any available
            concat_pred = pred_by_demo.get(test_demo_num_list[-1])
            if concat_pred is None and pred_by_demo:
                concat_pred = next(iter(pred_by_demo.values()))
            write_rollout_into_dict(result_dict, all_equations, None, concat_label, concat_data, test_demo_num_list, pred_by_demo=pred_by_demo)
        else:
            concat_pred = np.concatenate(all_preds, axis=2)
            concat_label = np.concatenate(all_labels, axis=2)
            concat_data = tree.tree_map(lambda *xs: np.concatenate(xs, axis=2), *all_data)
            write_rollout_into_dict(result_dict, all_equations, concat_pred, concat_label, concat_data, test_demo_num_list)
        total_batches_processed += 1

        if batch_idx == 0:
            first_batch_plot_data = (all_equations, all_captions, concat_pred, concat_label, concat_data)
            print('concat data (first batch): ', tree.tree_map(lambda x: x.shape, concat_data))

    # Plot using first batch only (to avoid too many files)
    if first_batch_plot_data is not None:
        all_equations, all_captions, concat_pred, concat_label, concat_data = first_batch_plot_data
        n_plot = min(2, len(all_equations) // FLAGS.num_repeat) if FLAGS.num_repeat else 1
        for fij in range(n_plot):
            start_i = FLAGS.num_repeat * fij
            end_i = start_i + FLAGS.num_repeat
            if end_i > len(all_equations):
                break
            batch_indices = slice(start_i, end_i)
            these_equations = [eq if isinstance(eq, str) else str(eq) for eq in all_equations[batch_indices]]
            these_captions = [cap if isinstance(cap, str) else str(cap) for cap in all_captions[batch_indices]]
            these_data = tree.tree_map(lambda x: x[batch_indices], concat_data)
            these_labels = concat_label[batch_indices]
            these_preds = concat_pred[batch_indices]
            if these_equations and FLAGS.correction:
                figure_sde_pathwise = plot.SDE_plot_pathwise_rollout(these_equations[0], these_captions,
                    tree.tree_map(lambda x: x[0], these_data), these_labels[0], these_preds[0], test_config, to_tfboard=False)
                figure_sde_pathwise.savefig(FLAGS.analysis_dir + "/rollout_pathwise_SDE_{}_{}.pdf".format(these_equations[0], fij))

    print(f"Rollout completed over {total_batches_processed} batch(es) (full dataset length).")
    for key, value in result_dict.items():
        result_dict[key] = np.array(value)
    
    # Print total inference time for rollout function
    print(f"\n{'='*60}")
    print(f"ROLLOUT TOTAL INFERENCE TIME: {total_inference_time:.4f} seconds")
    print(f"ROLLOUT TOTAL INFERENCE TIME: {total_inference_time/60:.2f} minutes")
    print(f"{'='*60}\n")
    
    return result_dict

def analyze_result_dict(result_dict):
	ground_truths = {}
	coarse_u = {}
	preds_demo_4 = {}
	# Per-demo_num: preds_by_demo[eqn][demo_num] = pred array (for logging error by number of demos)
	preds_by_demo = {}
	RMSE = {}
	MAE = {}
	std_MAE = {}
	rel_err = {}
	err_coarse = {}
	# No-correction case: model output directly vs ground truth qoi_v
	rel_err_no_corr = {}
	MAE_no_corr = {}
	RMSE_no_corr = {}
	std_MAE_no_corr = {}


	print("")
	for key, value in result_dict.items():
		print(key, value.shape, end = ", ", flush=True)
		print("")
		if "ground_truth" in key:
			if key[0] not in ground_truths:
				ground_truths[key[0]] = []
			ground_truths[key[0]].append(value)
		elif "cond_v" in key:
			if key[0] not in coarse_u:
				coarse_u[key[0]] = []
			coarse_u[key[0]].append(value)
		elif "pred" in key:
			if key[0] not in preds_demo_4:
				preds_demo_4[key[0]] = []
			preds_demo_4[key[0]].append(value)
			# Store by demo_num for per-demo error output
			if len(key) >= 3:
				demo_num = key[2]
				if key[0] not in preds_by_demo:
					preds_by_demo[key[0]] = {}
				preds_by_demo[key[0]][demo_num] = value


	for key, value in ground_truths.items():
		if ("ornstein_uhlenbeck" in key) or ("geombrownian_motion" in key) or ("ornsteinuhlenbeck" in key):
			qoi_k_Dflag = 1
		elif ("fluxgate" in key) or ("lorenz" in key) or ("predator_prey" in key):
			qoi_k_Dflag = 3
		else:
			qoi_k_Dflag = 2
		if FLAGS.correction:
			# No-correction: model output (pred) directly vs ground truth qoi_v (label + cond_v)
			gt_qoi_v = ground_truths[key][0] + coarse_u[key][0]
			pred_raw = preds_demo_4[key][0]
			if key not in rel_err_no_corr:
				rel_err_no_corr[key] = []
				MAE_no_corr[key] = []
				RMSE_no_corr[key] = []
				std_MAE_no_corr[key] = []
			rel_err_no_corr[key].append(np.mean(np.abs(pred_raw[:,:,:qoi_k_Dflag] - gt_qoi_v[:,:,:qoi_k_Dflag])/np.abs(gt_qoi_v[:,:,:qoi_k_Dflag])))
			MAE_no_corr[key].append(np.mean(np.abs(pred_raw[:,:,:qoi_k_Dflag] - gt_qoi_v[:,:,:qoi_k_Dflag])))
			RMSE_no_corr[key].append(np.sqrt(np.mean((pred_raw[:,:,:qoi_k_Dflag] - gt_qoi_v[:,:,:qoi_k_Dflag])**2)))
			std_MAE_no_corr[key].append(np.std(np.abs(pred_raw[:,:,:qoi_k_Dflag] - gt_qoi_v[:,:,:qoi_k_Dflag])))
			ground_truths[key] = ground_truths[key][0] + coarse_u[key][0]
			preds_demo_4[key] = preds_demo_4[key][0] + coarse_u[key][0]
		else:
			ground_truths[key] = ground_truths[key][0]
			preds_demo_4[key] = preds_demo_4[key][0]
			# No-correction same as only case: model output vs ground truth qoi_v
			if key not in rel_err_no_corr:
				rel_err_no_corr[key] = []
				MAE_no_corr[key] = []
				RMSE_no_corr[key] = []
				std_MAE_no_corr[key] = []

		print('ground truth shape: ', ground_truths[key].shape)

		if key not in rel_err:
			rel_err[key] = []
			err_coarse[key] = []
			RMSE[key] = []
			MAE[key] = []
			std_MAE[key] = []

		rel_err[key].append(np.mean(np.abs(preds_demo_4[key][:,:,:qoi_k_Dflag] - ground_truths[key][:,:,:qoi_k_Dflag])/np.abs(ground_truths[key][:,:,:qoi_k_Dflag])))
		MAE[key].append(np.mean(np.abs(preds_demo_4[key][:,:,:qoi_k_Dflag] - ground_truths[key][:,:,:qoi_k_Dflag])))
		RMSE[key].append(np.sqrt(np.mean((preds_demo_4[key][:,:,:qoi_k_Dflag] - ground_truths[key][:,:,:qoi_k_Dflag])**2)))
		std_MAE[key].append(np.std(np.abs(preds_demo_4[key][:,:,:qoi_k_Dflag] - ground_truths[key][:,:,:qoi_k_Dflag])))
		
		if FLAGS.correction:
			err_coarse[key].append(np.mean(np.abs(coarse_u[key][0][:,:,:qoi_k_Dflag] - ground_truths[key][:,:,:qoi_k_Dflag])))
			err_coarse[key].append(np.sqrt(np.mean((coarse_u[key][0][:,:,:qoi_k_Dflag] - ground_truths[key][:,:,:qoi_k_Dflag])**2)))
			err_coarse[key].append(np.std(np.abs(coarse_u[key][0][:,:,:qoi_k_Dflag] - ground_truths[key][:,:,:qoi_k_Dflag])))

		if not FLAGS.correction:
			rel_err_no_corr[key].append(np.mean(np.abs(preds_demo_4[key][:,:,:qoi_k_Dflag] - ground_truths[key][:,:,:qoi_k_Dflag])/np.abs(ground_truths[key][:,:,:qoi_k_Dflag])))
			MAE_no_corr[key].append(np.mean(np.abs(preds_demo_4[key][:,:,:qoi_k_Dflag] - ground_truths[key][:,:,:qoi_k_Dflag])))
			RMSE_no_corr[key].append(np.sqrt(np.mean((preds_demo_4[key][:,:,:qoi_k_Dflag] - ground_truths[key][:,:,:qoi_k_Dflag])**2)))
			std_MAE_no_corr[key].append(np.std(np.abs(preds_demo_4[key][:,:,:qoi_k_Dflag] - ground_truths[key][:,:,:qoi_k_Dflag])))

	# print('MAE: {}, std of MAE: {}, RMSE: {}'.format(MAE, std_MAE, RMSE))
	# # print('relative errors: ', rel_err)
	# if FLAGS.correction:
	# 	print('Errors for coarse u:, MAE, RMSE,std of MAE (in orders): ', err_coarse)
	print('Errors without correction (model output vs ground truth qoi_v): rel_err_no_corr={}, MAE_no_corr={}, RMSE_no_corr={}, std_MAE_no_corr={}'.format(rel_err_no_corr, MAE_no_corr, RMSE_no_corr, std_MAE_no_corr))

	# Output error by number of demos (for sweep_demo_nums or multiple test_demo_num_list)
	if preds_by_demo:
		print("", flush=True)
		print("========== Error by number of demos ==========", flush=True)
		for eqn in sorted(preds_by_demo.keys()):
			if eqn not in ground_truths:
				continue
			if ("ornstein_uhlenbeck" in eqn) or ("geombrownian_motion" in eqn) or ("ornsteinuhlenbeck" in eqn):
				qoi_k_Dflag = 1
			elif ("fluxgate" in eqn) or ("lorenz" in eqn) or ("predator_prey" in eqn):
				qoi_k_Dflag = 3
			else:
				qoi_k_Dflag = 2
			gt = ground_truths[eqn]
			coarse = coarse_u[eqn][0] if eqn in coarse_u and coarse_u[eqn] else None
			for demo_num in sorted(preds_by_demo[eqn].keys()):
				pred_arr = preds_by_demo[eqn][demo_num]
				if FLAGS.correction and coarse is not None:
					pred_full = pred_arr + coarse
				else:
					pred_full = pred_arr
				gt_slice = gt[:,:,:qoi_k_Dflag]
				pred_slice = pred_full[:,:,:qoi_k_Dflag]
				mae_d = np.mean(np.abs(pred_slice - gt_slice))
				rmse_d = np.sqrt(np.mean((pred_slice - gt_slice)**2))
				rel_d = np.mean(np.abs(pred_slice - gt_slice) / np.abs(gt_slice))
				print("  equation={}, num_demos={}: MAE={:.6e}, RMSE={:.6e}, rel_err={:.6e}".format(eqn, demo_num, mae_d, rmse_d, rel_d), flush=True)
		print("==============================================", flush=True)


def analyze_convg_result(convg_result_dict, FLAGS):
	ground_truths = {}
	coarse_u = {}
	preds_demo_4 = {}
	# Per-demo_num for logging error by number of demos
	preds_by_demo = {}
	MAE = {}
	RMSE = {}
	err_coarse = {}
	strong_convg = {}
	weak_convg = {}
	strong_coarse= {}
	weak_coarse = {}
	eqn = {}
	# No-correction case: model output directly vs ground truth qoi_v
	MAE_no_corr = {}
	RMSE_no_corr = {}
	strong_convg_no_corr = {}
	weak_convg_no_corr = {}

	print("")
	for key, value in convg_result_dict.items():
		print(key, value.shape, end = ", ", flush=True)
		print("")
		if "ground_truth" in key:
			if key[0] not in ground_truths:
				ground_truths[key[0]] = []
			ground_truths[key[0]].append(value)
		elif "cond_v" in key:
			if key[0] not in coarse_u:
				coarse_u[key[0]] = []
			coarse_u[key[0]].append(value)
		elif "pred" in key:
			if key[0] not in preds_demo_4:
				preds_demo_4[key[0]] = []
			preds_demo_4[key[0]].append(value)
			if len(key) >= 3:
				demo_num = key[2]
				if key[0] not in preds_by_demo:
					preds_by_demo[key[0]] = {}
				preds_by_demo[key[0]][demo_num] = value
		elif "equation" in key:
			if key[0] not in eqn:
				eqn[key[0]] = []
			eqn[key[0]].append(value)

	for key, value in ground_truths.items():
		if ("ornstein_uhlenbeck" in key) or ("geombrownian_motion" in key) or ("ornsteinuhlenbeck" in key):
			qoi_k_Dflag = 1
		elif ("fluxgate" in key) or ("lorenz" in key) or ("predator_prey" in key):
			qoi_k_Dflag = 3
		else:
			qoi_k_Dflag = 2
		# Rollout data: (n_samples, time_steps, dim) with time_steps > 50 (concatenated segments). No num_repeat grouping.
		is_rollout = (ground_truths[key][0].ndim >= 2 and ground_truths[key][0].shape[1] > 50)
		if is_rollout:
			# Errors over all samples and time steps; same formula as rollout plot: pred+coarse vs label+coarse
			gt_full = ground_truths[key][0] + coarse_u[key][0]
			pred_full = preds_demo_4[key][0] + coarse_u[key][0]
			gt_s = gt_full[:, :, :qoi_k_Dflag]
			pred_s = pred_full[:, :, :qoi_k_Dflag]
			if key not in strong_convg:
				strong_convg[key] = []
				weak_convg[key] = []
				strong_coarse[key] = []
				weak_coarse[key] = []
				MAE[key] = []
				RMSE[key] = []
				err_coarse[key] = []
				MAE_no_corr[key] = []
				RMSE_no_corr[key] = []
				strong_convg_no_corr[key] = []
				weak_convg_no_corr[key] = []
			MAE[key].append(np.mean(np.abs(pred_s - gt_s)))
			RMSE[key].append(np.sqrt(np.mean((pred_s - gt_s)**2)))
			MAE_no_corr[key].append(np.mean(np.abs(preds_demo_4[key][0][:,:,:qoi_k_Dflag] - gt_s)))
			RMSE_no_corr[key].append(np.sqrt(np.mean((preds_demo_4[key][0][:,:,:qoi_k_Dflag] - gt_s)**2)))
			strong_convg[key].append(np.mean(np.max(np.abs(pred_s - gt_s), axis=2)))
			weak_convg[key].append(np.mean(np.max(np.abs(np.mean(pred_s, axis=1) - np.mean(gt_s, axis=1)), axis=-1)))
			strong_convg_no_corr[key].append(np.mean(np.max(np.abs(preds_demo_4[key][0][:,:,:qoi_k_Dflag] - gt_s), axis=2)))
			weak_convg_no_corr[key].append(np.mean(np.max(np.abs(np.mean(preds_demo_4[key][0][:,:,:qoi_k_Dflag], axis=1) - np.mean(gt_s, axis=1)), axis=1)))
			if FLAGS.correction:
				coarse_s = coarse_u[key][0][:, :, :qoi_k_Dflag]
				mae_coarse = np.mean(np.abs(coarse_s - gt_s))
				rmse_coarse = np.sqrt(np.mean((coarse_s - gt_s)**2))
				err_coarse[key].append(mae_coarse)
				err_coarse[key].append(rmse_coarse)
				strong_coarse[key].append(np.mean(np.max(np.abs(coarse_s - gt_s), axis=2)))
				weak_coarse[key].append(np.mean(np.max(np.abs(np.mean(coarse_s, axis=1) - np.mean(gt_s, axis=1)), axis=1)))
			ground_truths[key] = gt_full
			preds_demo_4[key] = pred_full
			print('MAE: {}, RMSE: {} (rollout: all samples and time steps)'.format(MAE, RMSE))
			if FLAGS.correction:
				print('MAE, RMSE for coarse solution: ', err_coarse)
			print('Errors without correction: MAE_no_corr={}, RMSE_no_corr={}'.format(MAE_no_corr, RMSE_no_corr))
			continue

		if FLAGS.correction:
			# No-correction: model output (pred) directly vs ground truth qoi_v (label + cond_v)
			gt_qoi_v = ground_truths[key][0] + coarse_u[key][0]
			pred_raw = preds_demo_4[key][0]
			# Truncate to multiple of num_repeat so reshape is valid (e.g. 400 -> 396 when num_repeat=36)
			n0 = gt_qoi_v.shape[0]
			n_use = (n0 // FLAGS.num_repeat) * FLAGS.num_repeat
			if n_use < n0:
				gt_qoi_v = gt_qoi_v[-n_use:]
				pred_raw = pred_raw[-n_use:]
			new_shape_no_corr = (gt_qoi_v.shape[0] // FLAGS.num_repeat, FLAGS.num_repeat) + gt_qoi_v.shape[1:]
			ground_truths_repeat_no_corr = gt_qoi_v.reshape(new_shape_no_corr)
			preds_repeat_no_corr = pred_raw.reshape(new_shape_no_corr)
			if key not in MAE_no_corr:
				MAE_no_corr[key] = []
				RMSE_no_corr[key] = []
				strong_convg_no_corr[key] = []
				weak_convg_no_corr[key] = []
			MAE_no_corr[key].append(np.mean(np.abs(preds_repeat_no_corr[:,:,:qoi_k_Dflag] - ground_truths_repeat_no_corr[:,:,:qoi_k_Dflag])))
			RMSE_no_corr[key].append(np.sqrt(np.mean((preds_repeat_no_corr[:,:,:qoi_k_Dflag] - ground_truths_repeat_no_corr[:,:,:qoi_k_Dflag])**2)))
			strong_convg_no_corr[key].append(np.mean(np.max(np.abs(preds_repeat_no_corr[:,:,:qoi_k_Dflag] - ground_truths_repeat_no_corr[:,:,:qoi_k_Dflag]),axis=2)))
			weak_convg_no_corr[key].append(np.mean(np.max(np.abs(np.mean(preds_repeat_no_corr[:,:,:qoi_k_Dflag], axis=1) - np.mean(ground_truths_repeat_no_corr[:,:,:qoi_k_Dflag], axis=1)),axis=2)))
			# Use truncated arrays so downstream reshape (new_shape) is valid
			ground_truths[key] = gt_qoi_v
			preds_demo_4[key] = pred_raw
		else:
			gt = ground_truths[key][0]
			pred = preds_demo_4[key][0]
			n0 = gt.shape[0]
			n_use = (n0 // FLAGS.num_repeat) * FLAGS.num_repeat
			if n_use < n0:
				gt = gt[-n_use:]
				pred = pred[-n_use:]
			ground_truths[key] = gt
			preds_demo_4[key] = pred
			# No-correction same as only case
			if key not in MAE_no_corr:
				MAE_no_corr[key] = []
				RMSE_no_corr[key] = []
				strong_convg_no_corr[key] = []
				weak_convg_no_corr[key] = []

		# new shape (#, num_repeat, 50, x)
		new_shape = (ground_truths[key].shape[0] // FLAGS.num_repeat, FLAGS.num_repeat) + ground_truths[key].shape[1:]		
		ground_truths_repeat = ground_truths[key].reshape(new_shape)
		preds_repeat = preds_demo_4[key].reshape(new_shape)
		eqn_name = eqn[key][0]

		if key not in strong_convg:
			strong_convg[key] = []
			weak_convg[key] = []
			strong_coarse[key] = []
			weak_coarse[key] = []
			MAE[key] = []
			RMSE[key] = []
			err_coarse[key] = []

		
		MAE[key].append(np.mean(np.abs(preds_repeat[:,:,:qoi_k_Dflag] - ground_truths_repeat[:,:,:qoi_k_Dflag])))
		RMSE[key].append(np.sqrt(np.mean((preds_repeat[:,:,:qoi_k_Dflag] - ground_truths_repeat[:,:,:qoi_k_Dflag])**2)))
		# std_MAE[key].append(np.std(np.abs(preds_repeat[:,:,:qoi_k_Dflag] - ground_truths_repeat[:,:,:qoi_k_Dflag])))
		
		if FLAGS.correction:
			n_gt = ground_truths[key].shape[0]
			coarse_u_repeat = coarse_u[key][0][-n_gt:].reshape(new_shape)
			mae_coarse = np.mean(np.abs(coarse_u_repeat[:,:,:qoi_k_Dflag] - ground_truths_repeat[:,:,:qoi_k_Dflag]))
			rmse_coarse = np.sqrt(np.mean((coarse_u_repeat[:,:,:qoi_k_Dflag] - ground_truths_repeat[:,:,:qoi_k_Dflag])**2))
			err_coarse[key].append(mae_coarse)
			err_coarse[key].append(rmse_coarse)

		if not FLAGS.correction:
			MAE_no_corr[key].append(np.mean(np.abs(preds_repeat[:,:,:qoi_k_Dflag] - ground_truths_repeat[:,:,:qoi_k_Dflag])))
			RMSE_no_corr[key].append(np.sqrt(np.mean((preds_repeat[:,:,:qoi_k_Dflag] - ground_truths_repeat[:,:,:qoi_k_Dflag])**2)))
			strong_convg_no_corr[key].append(np.mean(np.max(np.abs(preds_repeat[:,:,:qoi_k_Dflag] - ground_truths_repeat[:,:,:qoi_k_Dflag]),axis=2)))
			weak_convg_no_corr[key].append(np.mean(np.max(np.abs(np.mean(preds_repeat[:,:,:qoi_k_Dflag], axis=1) - np.mean(ground_truths_repeat[:,:,:qoi_k_Dflag], axis=1)),axis=2)))

		print('MAE: {}, RMSE: {}'.format(MAE, RMSE))
		# print('relative errors: ', rel_err)
		if FLAGS.correction:
			print('MAE, RMSE for coarse soluion: ', err_coarse)
		print('Errors without correction (model output vs ground truth qoi_v): MAE_no_corr={}, RMSE_no_corr={}, strong_convg_no_corr={}, weak_convg_no_corr={}'.format(MAE_no_corr, RMSE_no_corr, strong_convg_no_corr, weak_convg_no_corr))

		strong = np.mean(np.max(np.abs(preds_repeat[:,:,:qoi_k_Dflag] - ground_truths_repeat[:,:,:qoi_k_Dflag]),axis=2))
		strong_convg[key].append(strong)
		weak = np.mean(np.max(np.abs(np.mean(preds_repeat[:,:,:qoi_k_Dflag], axis=1) - np.mean(ground_truths_repeat[:,:,:qoi_k_Dflag], axis=1)),axis=2))
		weak_convg[key].append(weak)


		if FLAGS.correction:
			coarse_u_repeat = coarse_u[key][0][-ground_truths[key].shape[0]:].reshape(new_shape)
			S_coarse = np.mean(np.max(np.abs(coarse_u_repeat[:,:,:qoi_k_Dflag] - ground_truths_repeat[:,:,:qoi_k_Dflag]),axis=2))
			W_coarse = np.mean(np.max(np.abs(np.mean(coarse_u_repeat[:,:,:qoi_k_Dflag], axis=1) - np.mean(ground_truths_repeat[:,:,:qoi_k_Dflag], axis=1)),axis=2))
			strong_coarse[key].append(S_coarse)
			weak_coarse[key].append(W_coarse)

			# Plotting only when FLAGS.correction is True; skipped otherwise
			error_dist = plot.error_distribution(preds_repeat, ground_truths_repeat, coarse_u_repeat, qoi_k_Dflag, to_tfboard = False)
			error_dist.savefig(FLAGS.analysis_dir+"/Error_dist_{}.pdf".format(key))
		


	# Output error by number of demos (for sweep_demo_nums or multiple test_demo_num_list)
	if preds_by_demo:
		print("", flush=True)
		print("========== Error by number of demos (convergence) ==========", flush=True)
		for eqn in sorted(preds_by_demo.keys()):
			if eqn not in ground_truths:
				continue
			if ("ornstein_uhlenbeck" in eqn) or ("geombrownian_motion" in eqn) or ("ornsteinuhlenbeck" in eqn):
				qoi_k_Dflag = 1
			elif ("fluxgate" in eqn) or ("lorenz" in eqn) or ("predator_prey" in eqn):
				qoi_k_Dflag = 3
			else:
				qoi_k_Dflag = 2
			gt = ground_truths[eqn]
			if isinstance(gt, list):
				gt = gt[0]
			coarse = coarse_u[eqn][0] if eqn in coarse_u and coarse_u[eqn] else None
			is_rollout_d = (gt.ndim >= 2 and gt.shape[1] > 50)
			if is_rollout_d:
				for demo_num in sorted(preds_by_demo[eqn].keys()):
					pred_arr = preds_by_demo[eqn][demo_num]
					pred_full = (pred_arr + coarse)[:gt.shape[0]] if (FLAGS.correction and coarse is not None) else pred_arr[:gt.shape[0]]
					gt_s = gt[:,:,:qoi_k_Dflag]
					pred_s = pred_full[:,:,:qoi_k_Dflag]
					mae_d = np.mean(np.abs(pred_s - gt_s))
					rmse_d = np.sqrt(np.mean((pred_s - gt_s)**2))
					strong_d = np.mean(np.max(np.abs(pred_s - gt_s), axis=2))
					weak_d = np.mean(np.max(np.abs(np.mean(pred_s, axis=1) - np.mean(gt_s, axis=1)), axis=-1))
					print("  equation={}, num_demos={}: MAE={:.6e}, RMSE={:.6e}, strong={:.6e}, weak={:.6e} (rollout)".format(eqn, demo_num, mae_d, rmse_d, strong_d, weak_d), flush=True)
			else:
				n_use_d = (gt.shape[0] // FLAGS.num_repeat) * FLAGS.num_repeat
				gt = gt[-n_use_d:]
				new_shape_d = (gt.shape[0] // FLAGS.num_repeat, FLAGS.num_repeat) + gt.shape[1:]
				gt_repeat = gt.reshape(new_shape_d)
				for demo_num in sorted(preds_by_demo[eqn].keys()):
					pred_arr = preds_by_demo[eqn][demo_num]
					if FLAGS.correction and coarse is not None:
						pred_full = (pred_arr + coarse)[-n_use_d:]
					else:
						pred_full = pred_arr[-n_use_d:]
					pred_repeat = pred_full.reshape(new_shape_d)
					gt_s = gt_repeat[:,:,:qoi_k_Dflag]
					pred_s = pred_repeat[:,:,:qoi_k_Dflag]
					mae_d = np.mean(np.abs(pred_s - gt_s))
					rmse_d = np.sqrt(np.mean((pred_s - gt_s)**2))
					strong_d = np.mean(np.max(np.abs(pred_s - gt_s), axis=2))
					weak_d = np.mean(np.max(np.abs(np.mean(pred_s, axis=1) - np.mean(gt_s, axis=1)), axis=2))
					print("  equation={}, num_demos={}: MAE={:.6e}, RMSE={:.6e}, strong={:.6e}, weak={:.6e}".format(eqn, demo_num, mae_d, rmse_d, strong_d, weak_d), flush=True)
		print("============================================================", flush=True)

	print('strong: {}, weak: {}'.format(strong_convg, weak_convg))
	print('Errors without correction (strong_convg_no_corr, weak_convg_no_corr): {}, {}'.format(strong_convg_no_corr, weak_convg_no_corr))
	if FLAGS.correction:
		print('strong for coarse u: {}, weak for coarse u: {}'.format(strong_coarse, weak_coarse))


def main(argv):
	for key, value in FLAGS.__flags.items():
		print(value.name, ": ", value._value, flush=True)

	tf.random.set_seed(FLAGS.seed + 123456) 
	result_dict = run_analysis()
	convg_result_dict = run_convergence()
	# rollout_result_dict = run_rollout()

	analyze_result_dict(result_dict)
	analyze_convg_result(convg_result_dict, FLAGS)
	# analyze_convg_result(rollout_result_dict, FLAGS)


	# if not os.path.exists(FLAGS.analysis_dir):
	# 	os.makedirs(FLAGS.analysis_dir)
	# with open("{}/result_dict{}.pkl".format(FLAGS.analysis_dir, FLAGS.results_name), "wb") as f:
	# 	pickle.dump(result_dict, f)

	# print("result_dict saved to {}".format(FLAGS.analysis_dir), flush=True)

if __name__ == '__main__':

	FLAGS = flags.FLAGS
	flags.DEFINE_boolean('tfboard', False, 'dump into tfboard')
	flags.DEFINE_boolean('correction', True, 'dump into tfboard')

	flags.DEFINE_enum('backend', 'torch', ['jax','torch'], 'backend of runner')

	flags.DEFINE_integer('seed', 42, 'random seed')
	flags.DEFINE_integer('num_repeat', 36, 'Different noise realization of the same initial value')

	flags.DEFINE_list('test_data_dirs', '/export/jyuan98/FMint_SDE/data_preparation', 'directories of testing data')
	flags.DEFINE_list('test_data_globs', ['test*'], 'filename glob patterns of testing data')
	flags.DEFINE_string('test_config_filename', 'test_input_id_config.json', 'config file for testing')
	flags.DEFINE_list('test_demo_num_list', [0,1,2,3,4], 'demo number list for testing')
	flags.DEFINE_boolean('sweep_demo_nums', False, 'if True, compute error with demo counts 1 to 4 (overrides test_demo_num_list)')
	flags.DEFINE_list('test_caption_id_list', [-1], 'caption id list for testing')

	flags.DEFINE_integer('batch_size', 128, 'batch size')
	flags.DEFINE_list('loss_mode', ['cap', 'nocap'], 'loss mode')
	# flags.DEFINE_list('loss_mode', ['nocap'], 'loss mode')
	flags.DEFINE_list('write', ["quest", "equation"], 'write mode')

	flags.DEFINE_string('model', 'gpt2', 'model name')
	flags.DEFINE_string('model_config_filename', '../config_model/model_gpt2_config.json', 'config file for model')
	flags.DEFINE_string('analysis_dir', '/export/users/song362/projects/in-context-operator-networks/icon-lm/analysis/', 'write file to dir')
	flags.DEFINE_string('results_name', '', 'additional file name for results')
	flags.DEFINE_string('restore_dir', '/home/shared/icon/save/user/ckpts/icon_gpt2/20230921-003808', 'restore directory')
	flags.DEFINE_integer('restore_step', 1000, 'restore step')


	app.run(main)
