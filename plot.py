import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import utils
import tensorflow as tf
import os

def get_plot_k_index(k_mode, equation):

	if ("ornstein_uhlenbeck" in equation) or ("geombrownian_motion" in equation) or ("ornsteinuhlenbeck" in equation):
		qoi_k_Dflag = 1
	elif ("fluxgate" in equation) or ("lorenz" in equation) or ("predator_prey" in equation):
		qoi_k_Dflag = 3
	else:
		qoi_k_Dflag = 2
	k_index = 0
	return k_index, qoi_k_Dflag


def plot_data(equation, caption, data, label, pred, config, to_tfboard = True):
	'''
	plot all figures in demo and prediction
	@param 
		equation: string
		caption: string
	@return
		the figure
	'''
	plt.close('all')
	k_index, qoi_k_Dflag = get_plot_k_index(config['k_mode'], equation)
	if qoi_k_Dflag == 2:
		fig, axs = plt.subplots(8, 1, figsize=(16, 12))
	elif qoi_k_Dflag == 3:
		fig, axs = plt.subplots(8, 1, figsize=(16, 12))
	else:
		fig, axs = plt.subplots(4, 1, figsize=(12, 8))
	fig.subplots_adjust(hspace=1.2, wspace=0.0)
	caption = ""
	fig.suptitle("eqn:{}\ncaption: {}".format(equation, caption))
	# plot cond quest
	
	# cond_quest = prompt[mask_cond_quest, :k_dim+v_dim]  # [cond_len_in_use, k_dim+v_dim]
	for i in range(len(data.demo_cond_mask)):
		axs[0].plot(data.demo_cond_k[i, data.demo_cond_mask[i,:].astype(bool), k_index], 
								data.demo_cond_v[i, data.demo_cond_mask[i,:].astype(bool), 0], 'o', markersize=3, label='u1, cond {}'.format(i), alpha = 0.5)
		if qoi_k_Dflag >= 2:
			axs[4].plot(data.demo_cond_k[i, data.demo_cond_mask[i,:].astype(bool), k_index], 
								data.demo_cond_v[i, data.demo_cond_mask[i,:].astype(bool), 1], 'o', markersize=3, label='u2, cond {}'.format(i), alpha = 0.5)
		
		axs[1].plot(data.demo_qoi_k[i, data.demo_qoi_mask[i,:].astype(bool), k_index],
								data.demo_qoi_v[i, data.demo_qoi_mask[i,:].astype(bool), 0], 'o', markersize=3, label='u1 qoi {}'.format(i), alpha = 0.5)
		if qoi_k_Dflag >= 2:
			axs[5].plot(data.demo_qoi_k[i, data.demo_qoi_mask[i,:].astype(bool), k_index],
								data.demo_qoi_v[i, data.demo_qoi_mask[i,:].astype(bool), 1], 'o', markersize=3, label='u2 qoi {}'.format(i), alpha = 0.5)
		
	axs[0].plot(data.quest_cond_k[0, data.quest_cond_mask[0,:].astype(bool), k_index], 
							data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0], 'k+', markersize=7, label='cond quest')
	if qoi_k_Dflag >= 2:
		axs[4].plot(data.quest_cond_k[0, data.quest_cond_mask[0,:].astype(bool), k_index], 
							data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1], 'k+', markersize=7, label='cond quest')
	
	axs[1].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							label[0, data.quest_qoi_mask[0,:].astype(bool), 0], 'k+', markersize=7, label='u1 qoi quest')
	if qoi_k_Dflag >= 2:
		axs[5].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							label[0, data.quest_qoi_mask[0,:].astype(bool), 1], 'k+', markersize=7, label='u2 qoi quest')
		
	# plot pred for both dimensions 
	axs[2].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							label[0, data.quest_qoi_mask[0,:].astype(bool), 0], 'k+', markersize=7, label='u1 qoi quest')
	axs[3].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							pred[data.quest_qoi_mask[0,:].astype(bool), 0], 'r+', markersize=7, label='u1 qoi pred')
	if qoi_k_Dflag >= 2:
		axs[6].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							label[0, data.quest_qoi_mask[0,:].astype(bool), 1], 'k+', markersize=7, label='u2 qoi quest')
		axs[7].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							pred[data.quest_qoi_mask[0,:].astype(bool), 1], 'r+', markersize=7, label='u2 qoi pred')

	demo_cond_len = np.sum(data.demo_cond_mask, axis=1)
	demo_qoi_len = np.sum(data.demo_qoi_mask, axis=1)
	quest_cond_len = np.sum(data.quest_cond_mask, axis=1)
	quest_qoi_len = np.sum(data.quest_qoi_mask, axis=1)

	axs[0].set_xlabel('key'); axs[0].set_ylabel('value')
	axs[1].set_xlabel('key'); axs[1].set_ylabel('value')
	axs[2].set_xlabel('key'); axs[2].set_ylabel('value')
	axs[3].set_xlabel('key'); axs[3].set_ylabel('value')
	axs[0].set_title("u1 cond, demo len: {}, quest len: {}".format(demo_cond_len, quest_cond_len))
	axs[1].set_title("u1 qoi, demo len: {}, quest len: {}".format(demo_qoi_len, quest_qoi_len))
	axs[2].set_title("u1 quest qoi ground truth, len: {}".format(quest_qoi_len))
	axs[3].set_title("u1 quest qoi prediction, len: {}".format(quest_qoi_len))

	if qoi_k_Dflag >= 2:
		axs[4].set_xlabel('key'); axs[3].set_ylabel('value')
		axs[5].set_xlabel('key'); axs[4].set_ylabel('value')
		axs[6].set_xlabel('key'); axs[5].set_ylabel('value')
		axs[7].set_xlabel('key'); axs[7].set_ylabel('value')

		axs[4].set_title("u2 cond, demo len: {}, quest len: {}".format(demo_cond_len, quest_cond_len))
		axs[5].set_title("u2 qoi, demo len: {}, quest len: {}".format(demo_qoi_len, quest_qoi_len))
		axs[6].set_title("u2 quest qoi ground truth, len: {}".format(quest_qoi_len))
		axs[7].set_title("u2 quest qoi prediction, len: {}".format(quest_qoi_len))
	if to_tfboard:
		return utils.plot_to_image(fig)
	else:  # save to a file
		return fig

def correction_plot_data(equation, caption, data, label, pred, config, to_tfboard = True):
	'''
	plot all figures in demo and prediction
	@param 
		equation: string
		caption: string
	@return
		the figure
	'''
	plt.close('all')
	k_index, qoi_k_Dflag = get_plot_k_index(config['k_mode'], equation)
	if qoi_k_Dflag == 3:
		fig, axs = plt.subplots(3, 1, figsize=(16, 12))
		fig.subplots_adjust(hspace=0.5, wspace=0.0)
	elif qoi_k_Dflag == 2:
		fig, axs = plt.subplots(2, 1, figsize=(16, 12))
		fig.subplots_adjust(hspace=0.5, wspace=0.0)
	else:
		fig, axs = plt.subplots(1, 1, figsize=(12, 8))
	
	caption = ""
	# print('quest qoi k: ', data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index])
	# fig.suptitle("eqn:{}\ncaption: {}".format(equation, caption))
	# plot pred for both dimensions 
	if qoi_k_Dflag >= 2:
		axs[0].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							label[0, data.quest_qoi_mask[0,:].astype(bool), 0], 'g+-', markersize=12, label='ground truth error')
		axs[0].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							pred[data.quest_qoi_mask[0,:].astype(bool), 0], 'co-', markersize=9, label='predicted error')
		axs[1].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							label[0, data.quest_qoi_mask[0,:].astype(bool), 1], 'g+-', markersize=12, label='ground truth error')
		axs[1].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							pred[data.quest_qoi_mask[0,:].astype(bool), 1], 'co-', markersize=9, label='predicted error')
		if qoi_k_Dflag == 3:
			axs[2].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							label[0, data.quest_qoi_mask[0,:].astype(bool), 2], 'g+-', markersize=12, label='ground truth error')
			axs[2].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							pred[data.quest_qoi_mask[0,:].astype(bool), 2], 'co-', markersize=9, label='predicted error')
		
	else:
		axs.plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							label[0, data.quest_qoi_mask[0,:].astype(bool), 0], 'g+-', markersize=12, label='ground truth error')
		axs.plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							pred[data.quest_qoi_mask[0,:].astype(bool), 0], 'co-', markersize=9, label='predicted error')

	if qoi_k_Dflag >= 2:
		axs[0].set_xlabel('time')
		axs[0].legend(loc = "upper right", fontsize="25")
		axs[0].yaxis.set_major_locator(ticker.LinearLocator(4))
		axs[0].yaxis.set_minor_locator(ticker.LinearLocator(8))
		axs[1].set_xlabel('time')
		axs[1].legend(loc = "upper right", fontsize="25")
		axs[1].yaxis.set_major_locator(ticker.LinearLocator(4))
		axs[1].yaxis.set_minor_locator(ticker.LinearLocator(8))
		if qoi_k_Dflag == 3:
			axs[2].set_xlabel('time')
			axs[2].legend(loc = "upper right", fontsize="25")
			axs[2].yaxis.set_major_locator(ticker.LinearLocator(4))
			axs[2].yaxis.set_minor_locator(ticker.LinearLocator(8))
			for item in ([axs[2].xaxis.label, axs[2].yaxis.label] + axs[2].get_xticklabels() + axs[2].get_yticklabels()):
				item.set_fontsize(25)

		for item in ([axs[0].xaxis.label, axs[0].yaxis.label] + axs[0].get_xticklabels() + axs[0].get_yticklabels()):
			item.set_fontsize(25)
		for item in ([axs[1].xaxis.label, axs[1].yaxis.label] + axs[1].get_xticklabels() + axs[1].get_yticklabels()):
			item.set_fontsize(25)
	else:
		axs.set_xlabel('time')
		axs.legend(loc = "upper right",fontsize="25")

		axs.yaxis.set_major_locator(ticker.LinearLocator(4))
		axs.yaxis.set_minor_locator(ticker.LinearLocator(8))

		for item in ([axs.xaxis.label, axs.yaxis.label] + axs.get_xticklabels() + axs.get_yticklabels()):
			item.set_fontsize(25)
	
	if to_tfboard:
		return utils.plot_to_image(fig)
	else:  # save to a file
		return fig
	
def SDE_plot_pathwise(equation, caption, data, label, pred, config, to_tfboard = True):
	'''
	plot all figures in demo and prediction
	@param 
		equation: string
		caption: string
	@return
		the figure
	'''
	plt.close('all')
	k_index, qoi_k_Dflag = get_plot_k_index(config['k_mode'], equation)
	if qoi_k_Dflag == 3:
		fig, axs = plt.subplots(3, 1, figsize=(16, 12))
		fig.subplots_adjust(hspace=0.4, wspace=0.0)
	elif qoi_k_Dflag == 2:
		fig, axs = plt.subplots(2, 1, figsize=(16, 12))
		fig.subplots_adjust(hspace=0.4, wspace=0.0)
	else:
		fig, axs = plt.subplots(1, 1, figsize=(12, 8))
	
	caption = ""

	ind_cond = np.argsort(data.quest_cond_k[0, data.quest_cond_mask[0,:].astype(bool), 0])
	ind_qoi = np.argsort(data.quest_qoi_k[0, data.quest_cond_mask[0,:].astype(bool), 0])

	# print('label shape in pathwise: ', label.shape)

	if qoi_k_Dflag >= 2:
		fine_u_0 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]
		fine_u_1 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond]

		# FMint-SDE prediction (fine correction added to coarse)
		fmint_u_0 = (pred[data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]
		fmint_u_1 = (pred[data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond]

		# Coarse solution directly from quest_cond_v (already time-aligned)
		coarse_0 = (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]
		coarse_1 = (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond]

		x_vals = np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index])

		axs[0].plot(x_vals, fine_u_0, 'r+-', markersize=15, label='Fine')
		axs[0].plot(x_vals, coarse_0, 'g+-', markersize=15, label='Coarse')
		axs[0].plot(x_vals, fmint_u_0, 'o-', markersize=9, label='FMint-SDE')

		axs[1].plot(x_vals, fine_u_1, 'r+-', markersize=15, label='Fine')
		axs[1].plot(x_vals, coarse_1, 'g+-', markersize=15, label='Coarse')
		axs[1].plot(x_vals, fmint_u_1, 'o-', markersize=9, label='FMint-SDE')
		if qoi_k_Dflag == 3:
			fine_u_2 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 2])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 2])[ind_cond]
			fmint_u_2 = (pred[data.quest_qoi_mask[0,:].astype(bool), 2])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 2])[ind_cond]
			coarse_2 = (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 2])[ind_cond]

			x_vals = np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index])

			axs[2].plot(x_vals, fine_u_2, 'r+-', markersize=15, label='Fine')
			axs[2].plot(x_vals, coarse_2, 'g+-', markersize=15, label='Coarse')
			axs[2].plot(x_vals, fmint_u_2, 'o-', markersize=9, label='FMint-SDE')

	else:
		fine_u = (label[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]

		fmint_u = (pred[data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]

		coarse = (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]

		x_vals = np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index])

		axs.plot(x_vals,
							fine_u, 'r+-', markersize=15, label='Fine')
		axs.plot(x_vals,
							coarse, 'g+-', markersize=15, label='Coarse')
		axs.plot(x_vals,
							fmint_u, 'o-', markersize=9, label='FMint-SDE', alpha = 0.5)

	if qoi_k_Dflag >= 2:
		axs[0].set_xlabel('time')
		axs[0].set_ylabel('value')
		axs[0].legend(loc = "upper right", fontsize="25")
		axs[0].yaxis.set_major_locator(ticker.LinearLocator(4))
		axs[0].yaxis.set_minor_locator(ticker.LinearLocator(8))
		axs[1].set_xlabel('time')
		axs[1].set_ylabel('value')
		axs[1].legend(loc = "upper right",fontsize="25")
		axs[1].yaxis.set_major_locator(ticker.LinearLocator(4))
		axs[1].yaxis.set_minor_locator(ticker.LinearLocator(8))
		if qoi_k_Dflag == 3:
			axs[2].set_xlabel('time')
			axs[2].set_ylabel('value')
			axs[2].legend(loc = "upper right",fontsize="25")
			axs[2].yaxis.set_major_locator(ticker.LinearLocator(4))
			axs[2].yaxis.set_minor_locator(ticker.LinearLocator(8))
			for item in ([axs[2].xaxis.label, axs[2].yaxis.label] + axs[2].get_xticklabels() + axs[2].get_yticklabels()):
				item.set_fontsize(25)

		for item in ([axs[0].xaxis.label, axs[0].yaxis.label] + axs[0].get_xticklabels() + axs[0].get_yticklabels()):
			item.set_fontsize(25)
		for item in ([axs[1].xaxis.label, axs[1].yaxis.label] + axs[1].get_xticklabels() + axs[1].get_yticklabels()):
			item.set_fontsize(25)
	else:
		axs.set_xlabel('time'); axs.set_ylabel('value')
		axs.legend(loc = "upper right", fontsize="25")

		axs.yaxis.set_major_locator(ticker.LinearLocator(4))
		axs.yaxis.set_minor_locator(ticker.LinearLocator(8))

		for item in ([axs.xaxis.label, axs.yaxis.label] + axs.get_xticklabels() + axs.get_yticklabels()):
			item.set_fontsize(25)
	
	if to_tfboard:
		return utils.plot_to_image(fig)
	else:  # save to a file
		return fig

def SDE_plot_pathwise_rollout(equation, caption, data, label, pred, config, to_tfboard = True):
	'''
	plot all figures in demo and prediction
	@param 
		equation: string
		caption: string
	@return
		the figure
	'''
	plt.close('all')
	k_index, qoi_k_Dflag = get_plot_k_index(config['k_mode'], equation)
	if qoi_k_Dflag == 3:
		fig, axs = plt.subplots(3, 1, figsize=(16, 12))
		fig.subplots_adjust(hspace=0.4, wspace=0.0)
	elif qoi_k_Dflag == 2:
		fig, axs = plt.subplots(2, 1, figsize=(16, 12))
		fig.subplots_adjust(hspace=0.4, wspace=0.0)
	else:
		fig, axs = plt.subplots(1, 1, figsize=(12, 8))
	
	caption = ""

	# print('rollout time: ', np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]))

	ind_cond = np.argsort(data.quest_cond_k[0, data.quest_cond_mask[0,:].astype(bool), 0])
	ind_qoi = np.argsort(data.quest_qoi_k[0, data.quest_cond_mask[0,:].astype(bool), 0])

	if qoi_k_Dflag >= 2:
		fine_u_0 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]
		fine_u_1 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond]

		fmint_u_0 = (pred[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]
		fmint_u_1 = (pred[0, data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond]

		coarse_0 = (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]
		coarse_1 = (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond]

		x_vals = np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index])

		axs[0].plot(x_vals, fine_u_0, 'r+-', markersize=15, label='Fine')
		axs[0].plot(x_vals, coarse_0, 'g+-', markersize=15, label='Coarse')
		axs[0].plot(x_vals, fmint_u_0, 'o-', markersize=9, label='FMint-SDE')

		axs[1].plot(x_vals, fine_u_1, 'r+-', markersize=15, label='Fine')
		axs[1].plot(x_vals, coarse_1, 'g+-', markersize=15, label='Coarse')
		axs[1].plot(x_vals, fmint_u_1, 'o-', markersize=9, label='FMint-SDE')
		if qoi_k_Dflag == 3:
			fine_u_2 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 2])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 2])[ind_cond]
			fmint_u_2 = (pred[0, data.quest_qoi_mask[0,:].astype(bool), 2])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 2])[ind_cond]
			coarse_2 = (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 2])[ind_cond]

			x_vals = np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index])

			axs[2].plot(x_vals, fine_u_2, 'r+-', markersize=15, label='Fine')
			axs[2].plot(x_vals, coarse_2, 'g+-', markersize=15, label='Coarse')
			axs[2].plot(x_vals, fmint_u_2, 'o-', markersize=9, label='FMint-SDE')

	else:
		fine_u = (label[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]

		fmint_u = (pred[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]

		coarse = (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]

		x_vals = np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index])

		axs.plot(x_vals,
							fine_u, 'r+-', markersize=15, label='Fine')
		axs.plot(x_vals,
							coarse, 'g+-', markersize=15, label='Coarse')
		axs.plot(x_vals,
							fmint_u, 'o-', markersize=9, label='FMint-SDE', alpha = 0.5)

	if qoi_k_Dflag >= 2:
		axs[0].set_xlabel('time')
		axs[0].set_ylabel('value')
		axs[0].legend(loc = "upper right", fontsize="25")
		axs[0].yaxis.set_major_locator(ticker.LinearLocator(4))
		axs[0].yaxis.set_minor_locator(ticker.LinearLocator(8))
		axs[1].set_xlabel('time')
		axs[1].set_ylabel('value')
		axs[1].legend(loc = "upper right",fontsize="25")
		axs[1].yaxis.set_major_locator(ticker.LinearLocator(4))
		axs[1].yaxis.set_minor_locator(ticker.LinearLocator(8))
		if qoi_k_Dflag == 3:
			axs[2].set_xlabel('time')
			axs[2].set_ylabel('value')
			axs[2].legend(loc = "upper right",fontsize="25")
			axs[2].yaxis.set_major_locator(ticker.LinearLocator(4))
			axs[2].yaxis.set_minor_locator(ticker.LinearLocator(8))
			for item in ([axs[2].xaxis.label, axs[2].yaxis.label] + axs[2].get_xticklabels() + axs[2].get_yticklabels()):
				item.set_fontsize(25)

		for item in ([axs[0].xaxis.label, axs[0].yaxis.label] + axs[0].get_xticklabels() + axs[0].get_yticklabels()):
			item.set_fontsize(25)
		for item in ([axs[1].xaxis.label, axs[1].yaxis.label] + axs[1].get_xticklabels() + axs[1].get_yticklabels()):
			item.set_fontsize(25)
	else:
		axs.set_xlabel('time'); axs.set_ylabel('value')
		axs.legend(loc = "upper right", fontsize="25")

		axs.yaxis.set_major_locator(ticker.LinearLocator(4))
		axs.yaxis.set_minor_locator(ticker.LinearLocator(8))

		for item in ([axs.xaxis.label, axs.yaxis.label] + axs.get_xticklabels() + axs.get_yticklabels()):
			item.set_fontsize(25)
	
	if to_tfboard:
		return utils.plot_to_image(fig)
	else:  # save to a file
		return fig
	
# def SDE_plot_expectation(equation, caption, data, label, pred, config, num_repeat, to_tfboard = True):
# 	'''
# 	plot all figures in demo and prediction
# 	@param 
# 		equation: string
# 		caption: string
# 	@return
# 		the figure
# 	'''
# 	plt.close('all')
# 	k_index, qoi_k_Dflag = get_plot_k_index(config['k_mode'], equation)
# 	if qoi_k_Dflag == 3:
# 		fig, axs = plt.subplots(3, 2, figsize=(32, 16))
# 		fig.subplots_adjust(hspace=0.4, wspace=0.0)
# 	elif qoi_k_Dflag == 2:
# 		fig, axs = plt.subplots(2, 2, figsize=(32, 16))
# 		fig.subplots_adjust(hspace=0.4, wspace=0.0)
# 	else:
# 		fig, axs = plt.subplots(1, 2, figsize=(32, 12))
	
# 	caption = ""

# 	fine_u_0_list = []
# 	fine_u_1_list = []
# 	fine_u_2_list = []

# 	print('equations in expectation: ', equation)

# 	for i in range(num_repeat):  # loop over batch
# 		ind_cond = np.argsort(data.quest_cond_k[i, data.quest_cond_mask[i, :].astype(bool), 0])
# 		ind_qoi = np.argsort(data.quest_qoi_k[i, data.quest_qoi_mask[i, :].astype(bool), 0])

# 		if qoi_k_Dflag >= 2:
# 			fine_u_0 = (label[i, data.quest_qoi_mask[i, :].astype(bool), 0])[ind_qoi] + \
# 					(data.quest_cond_v[i, data.quest_cond_mask[i, :].astype(bool), 0])[ind_cond]
# 			fine_u_1 = (label[i, data.quest_qoi_mask[i, :].astype(bool), 1])[ind_qoi] + \
# 					(data.quest_cond_v[i, data.quest_cond_mask[i, :].astype(bool), 1])[ind_cond]
# 			pred_u_0 = (pred[i, data.quest_qoi_mask[i, :].astype(bool), 0])[ind_qoi] + \
# 						(data.quest_cond_v[i, data.quest_cond_mask[i, :].astype(bool), 0])[ind_cond]
# 			pred_u_1 = (pred[i, data.quest_qoi_mask[i, :].astype(bool), 1])[ind_qoi] + \
# 						(data.quest_cond_v[i, data.quest_cond_mask[i, :].astype(bool), 1])[ind_cond]

# 			x_vals = np.sort(data.quest_qoi_k[i, data.quest_qoi_mask[i, :].astype(bool), k_index])

# 			# axs[0].plot(x_vals, fine_u_0, color + '+-', markersize=10, alpha=0.6)
# 			axs[0,0].plot(x_vals, (data.quest_cond_v[i, data.quest_cond_mask[i, :].astype(bool), 0])[ind_cond], 'x-', markersize=8, alpha=0.6)
# 			axs[0,1].plot(x_vals, pred_u_0, 'x-', markersize=8, alpha=0.6)

# 			# axs[1].plot(x_vals, fine_u_1, color + '+-', markersize=10, alpha=0.6)
# 			axs[1,0].plot(x_vals, (data.quest_cond_v[i, data.quest_cond_mask[i, :].astype(bool), 1])[ind_cond], 'x-', markersize=8, alpha=0.6)
# 			axs[1,1].plot(x_vals, pred_u_1, 'x-', markersize=8, alpha=0.6)

# 			fine_u_0_list.append(fine_u_0)
# 			fine_u_1_list.append(fine_u_1)

# 			if qoi_k_Dflag == 3:
# 				fine_u_2 = (label[i, data.quest_qoi_mask[i, :].astype(bool), 2])[ind_qoi] + \
# 						(data.quest_cond_v[i, data.quest_cond_mask[i, :].astype(bool), 2])[ind_cond]
# 				pred_u_2 = (pred[i, data.quest_qoi_mask[i, :].astype(bool), 2])[ind_qoi] + \
# 							(data.quest_cond_v[i, data.quest_cond_mask[i, :].astype(bool), 2])[ind_cond]
# 				# axs[2].plot(x_vals, fine_u_2, color + '+-', markersize=10, alpha=0.6)
# 				axs[2,0].plot(x_vals, (data.quest_cond_v[i, data.quest_cond_mask[i, :].astype(bool), 2])[ind_cond], 'x-', markersize=8, alpha=0.6)
# 				axs[2,1].plot(x_vals, pred_u_2, 'x-', markersize=8, alpha=0.6)

# 				fine_u_2_list.append(fine_u_2)

# 		else:
# 			fine_u = (label[i, data.quest_qoi_mask[i, :].astype(bool), 0])[ind_qoi] + \
# 					(data.quest_cond_v[i, data.quest_cond_mask[i, :].astype(bool), 0])[ind_cond]
# 			pred_u = (pred[i, data.quest_qoi_mask[i, :].astype(bool), 0])[ind_qoi] + \
# 					(data.quest_cond_v[i, data.quest_cond_mask[i, :].astype(bool), 0])[ind_cond]
# 			x_vals = np.sort(data.quest_qoi_k[i, data.quest_qoi_mask[i, :].astype(bool), k_index])
# 			# axs.plot(x_vals, fine_u, color + '+-', markersize=10, alpha=0.6)
# 			axs[0].plot(x_vals, (data.quest_cond_v[i, data.quest_cond_mask[i, :].astype(bool), 0])[ind_cond], 'x-', markersize=8, alpha=0.6)
# 			axs[1].plot(x_vals, pred_u, 'x-', markersize=8, alpha=0.6)

# 			fine_u_0_list.append(fine_u)
	
# 	fine_u_stack_0 = np.stack(fine_u_0_list, axis=0)  # shape: (batch_size, time_steps)
# 	fine_u_mean_0 = np.mean(fine_u_stack_0, axis=0)   # shape: (time_steps,)
# 	if qoi_k_Dflag >= 2:
# 		fine_u_stack_1 = np.stack(fine_u_1_list, axis=0)  # shape: (batch_size, time_steps)
# 		fine_u_mean_1 = np.mean(fine_u_stack_1, axis=0)   # shape: (time_steps,)
# 		if qoi_k_Dflag >= 3:
# 			fine_u_stack_2 = np.stack(fine_u_2_list, axis=0)  # shape: (batch_size, time_steps)
# 			fine_u_mean_2 = np.mean(fine_u_stack_2, axis=0)   # shape: (time_steps,)

# 	# Use any i to get the sorted time values (assuming all examples are aligned in time)
# 	x_vals = np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0, :].astype(bool), k_index])

# 	# Plot
# 	if qoi_k_Dflag == 1:
# 		axs[0].plot(x_vals, fine_u_mean_0, 'r-', linewidth=3, label='Avg Fine')
# 		axs[1].plot(x_vals, fine_u_mean_0, 'r-', linewidth=3, label='Avg Fine')
# 	else:
# 		axs[0,0].plot(x_vals, fine_u_mean_0, 'r-', linewidth=3, label='Avg Fine')
# 		axs[0,1].plot(x_vals, fine_u_mean_0, 'r-', linewidth=3, label='Avg Fine')
# 		axs[1,0].plot(x_vals, fine_u_mean_1, 'r-', linewidth=3, label='Avg Fine')
# 		axs[1,1].plot(x_vals, fine_u_mean_1, 'r-', linewidth=3, label='Avg Fine')
# 		if qoi_k_Dflag == 3:
# 			axs[2,0].plot(x_vals, fine_u_mean_2, 'r-', linewidth=3, label='Avg Fine')
# 			axs[2,1].plot(x_vals, fine_u_mean_2, 'r-', linewidth=3, label='Avg Fine')


# 	if qoi_k_Dflag >= 2:
# 		axs = np.atleast_2d(axs)  # ensure axs is always 2D

# 		# Set titles at the top of each column
# 		axs[0, 0].set_title('Coarse', fontsize=28)
# 		axs[0, 1].set_title('FMint-SDE', fontsize=28)

# 		# Set up plots
# 		for i in range(qoi_k_Dflag):
# 			for j in range(2):
# 				axs[i, j].set_xlabel('time')
# 				axs[i, j].set_ylabel('value')
# 				axs[i, j].legend(loc="upper right", fontsize=20)
# 				axs[i, j].yaxis.set_major_locator(ticker.LinearLocator(4))
# 				axs[i, j].yaxis.set_minor_locator(ticker.LinearLocator(8))

# 				# Set font sizes
# 				for item in ([axs[i, j].xaxis.label, axs[i, j].yaxis.label] +
# 							axs[i, j].get_xticklabels() + axs[i, j].get_yticklabels()):
# 					item.set_fontsize(20)
# 	else:
# 		for j in range(2):
# 			axs[j].set_xlabel('time')
# 			axs[j].set_ylabel('value')
# 			axs[j].legend(loc="upper right", fontsize=25)
# 			axs[j].yaxis.set_major_locator(ticker.LinearLocator(4))
# 			axs[j].yaxis.set_minor_locator(ticker.LinearLocator(8))

# 			for item in ([axs[j].xaxis.label, axs[j].yaxis.label] +
# 						axs[j].get_xticklabels() + axs[j].get_yticklabels()):
# 				item.set_fontsize(25)

# 	if qoi_k_Dflag == 1:
# 		ymin0, ymax0 = axs[0].get_ylim()
# 		ymin1, ymax1 = axs[1].get_ylim()
# 		ymin = min(ymin0, ymin1)
# 		ymax = max(ymax0, ymax1)
# 		axs[0].set_ylim(ymin, ymax)
# 		axs[1].set_ylim(ymin, ymax)
# 	else:
# 		# Set shared y-axis limits per column
# 		for i in range(qoi_k_Dflag):  # for each row
# 			ymins, ymaxs = [], []
# 			for j in range(2):
# 				ymin, ymax = axs[i, j].get_ylim()
# 				ymins.append(ymin)
# 				ymaxs.append(ymax)
# 			shared_ymin = min(ymins)
# 			shared_ymax = max(ymaxs)
# 			for j in range(2):
# 				axs[i, j].set_ylim(shared_ymin, shared_ymax)

# 	fig.tight_layout()
# 	if to_tfboard:
# 		return utils.plot_to_image(fig)
# 	else:  # save to a file
# 		return fig

def SDE_plot_expectation(equation, caption, data, label, pred, config, num_repeat, to_tfboard=True):
    """
    Plot expectation (across num_repeat) for coarse, prediction, and fine solutions.
    - One subplot per QoI component (qoi_k_Dflag in {1,2,3}).
    - Each subplot shows exactly three curves: Avg Coarse, Avg FMint-SDE, Avg Fine.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    plt.close('all')

    # Determine which k to plot and how many components (QoI dimensions)
    k_index, qoi_k_Dflag = get_plot_k_index(config['k_mode'], equation)

    # Create figure with one column and qoi_k_Dflag rows
    nrows = qoi_k_Dflag if qoi_k_Dflag in (2, 3) else 1
    fig, axs = plt.subplots(nrows, 1, figsize=(16, 5 * nrows))
    if nrows == 1:
        axs = np.atleast_1d(axs)

    # Storage for averages
    coarse_lists = [[] for _ in range(qoi_k_Dflag)]
    fine_lists   = [[] for _ in range(qoi_k_Dflag)]
    pred_lists   = [[] for _ in range(qoi_k_Dflag)]

    # Use first sample for x-axis grid (assumed consistent)
    first_ind_qoi = np.argsort(data.quest_qoi_k[0, data.quest_qoi_mask[0, :].astype(bool), 0])
    x_vals = np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0, :].astype(bool), k_index])

    # Collect runs
    for i in range(num_repeat):
        ind_cond = np.argsort(data.quest_cond_k[i, data.quest_cond_mask[i, :].astype(bool), 0])
        ind_qoi  = np.argsort(data.quest_qoi_k[i, data.quest_qoi_mask[i, :].astype(bool), 0])

        for comp in range(qoi_k_Dflag):
            coarse = (data.quest_cond_v[i, data.quest_cond_mask[i, :].astype(bool), comp])[ind_cond]
            fine   = (label[i, data.quest_qoi_mask[i, :].astype(bool), comp])[ind_qoi] + coarse
            pred_u = (pred[i, data.quest_qoi_mask[i, :].astype(bool), comp])[ind_qoi] + coarse

            coarse_lists[comp].append(coarse)
            fine_lists[comp].append(fine)
            pred_lists[comp].append(pred_u)

    # Compute averages and plot
    for comp in range(qoi_k_Dflag):
        ax = axs[comp] if qoi_k_Dflag > 1 else axs[0]

        coarse_mean = np.mean(np.stack(coarse_lists[comp], axis=0), axis=0)
        fine_mean   = np.mean(np.stack(fine_lists[comp],   axis=0), axis=0)
        pred_mean   = np.mean(np.stack(pred_lists[comp],   axis=0), axis=0)

        # --- Plot lines (thicker) ---
        ax.plot(x_vals, coarse_mean, '-',  linewidth=3.5, label='Avg Coarse')
        ax.plot(x_vals, pred_mean,   '--', linewidth=3.5, label='Avg FMint-SDE')
        ax.plot(x_vals, fine_mean,   '-.', linewidth=3.5, label='Avg Fine')

        # Axis labels and legend (larger fonts)
        ax.set_xlabel('time', fontsize=20)
        ax.set_ylabel('value', fontsize=20)
        ax.legend(loc='best', fontsize=20, frameon=True)

        # Tick settings
        ax.yaxis.set_major_locator(ticker.LinearLocator(4))
        ax.yaxis.set_minor_locator(ticker.LinearLocator(8))
        ax.tick_params(axis='both', which='major', labelsize=18, width=1.5)
        ax.tick_params(axis='both', which='minor', labelsize=16, width=1.5)

        # Thicker axis spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # Title with larger font
    fig.suptitle(caption if caption else "", fontsize=22, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Output
    if to_tfboard:
        return utils.plot_to_image(fig)
    else:
        return fig



# def error_distribution(preds, ground_truths, coarse_u, qoi_k_Dflag, to_tfboard = False):
	# plt.close('all')
	# fig, axs = plt.subplots(1, 2, figsize=(12, 8))

	# # Compute errors
	# err_preds = preds[:, :, :qoi_k_Dflag] - ground_truths[:, :, :qoi_k_Dflag]
	# err_coarse = coarse_u[:, :, :qoi_k_Dflag] - ground_truths[:, :, :qoi_k_Dflag]

	# # Flatten for histogram
	# err_preds_flat = err_preds.flatten()
	# err_coarse_flat = err_coarse.flatten()

	# # Compute shared bin edges for same x-axis
	# all_errors = np.concatenate([err_preds_flat, err_coarse_flat])
	# bin_edges = np.histogram_bin_edges(all_errors, bins=50)

	# # Plot histograms as percentages
	# hist_preds, _, _ = axs[0].hist(err_preds_flat, bins=bin_edges, alpha=0.5, density=True)
	# hist_coarse, _, _ = axs[1].hist(err_coarse_flat, bins=bin_edges, alpha=0.5, density=True)

    # # Determine the maximum y value across both histograms (as density), then scale to percentage
	# max_density = max(hist_preds.max(), hist_coarse.max())
	# max_percent = max_density * 100
	# max_percent_rounded = np.ceil(max_percent / 10) * 10  # round to nearest 10 for cleaner axis

    # # Convert y-axis from density to percentage and set fixed limits
	# for ax in axs:
	# 	y_ticks = np.linspace(0, max_percent_rounded, num=6)  # evenly spaced ticks
	# 	ax.set_ylim(0, max_percent_rounded / 100)
	# 	ax.set_yticks(y_ticks / 100)
	# 	ax.set_yticklabels([f"{y:.0f}%" for y in y_ticks])

    # # Titles and labels
	# axs[0].set_title('FMint-SDE', fontsize=24)
	# axs[1].set_title('Coarse Simulation', fontsize=24)
	# axs[0].set_xlabel('Error', fontsize=20)
	# axs[0].set_ylabel('Percentage', fontsize=20)
	# axs[1].set_xlabel('Error', fontsize=20)
	# axs[1].set_ylabel('Percentage', fontsize=20)

    # # Format ticks and spines
	# for ax in axs:
	# 	ax.spines['left'].set_position(('data', 0))
	# 	ax.spines['right'].set_color('none')
	# 	ax.spines['top'].set_color('none')
	# 	ax.yaxis.set_ticks_position('left')
	# 	ax.xaxis.set_ticks_position('bottom')
	# 	ax.tick_params(axis='both', which='major', labelsize=16)
	
	# fig.tight_layout()
	
	# if to_tfboard:
	# 	return utils.plot_to_image(fig)
	# else:
	# 	return fig

def error_distribution(preds, ground_truths, coarse_u, qoi_k_Dflag, to_tfboard=False):
    plt.close('all')
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    # Compute errors
    err_preds = preds[:, :, :qoi_k_Dflag] - ground_truths[:, :, :qoi_k_Dflag]
    err_coarse = coarse_u[:, :, :qoi_k_Dflag] - ground_truths[:, :, :qoi_k_Dflag]

    # Flatten for histogram
    err_preds_flat = err_preds.flatten()
    err_coarse_flat = err_coarse.flatten()

    # Shared bin edges
    all_errors = np.concatenate([err_preds_flat, err_coarse_flat])
    bin_edges = np.histogram_bin_edges(all_errors, bins=50)
    bin_widths = np.diff(bin_edges)

    # Compute histograms manually as percentages
    hist_preds, _ = np.histogram(err_preds_flat, bins=bin_edges)
    hist_coarse, _ = np.histogram(err_coarse_flat, bins=bin_edges)

    # Convert counts to percentage
    total_preds = len(err_preds_flat)
    total_coarse = len(err_coarse_flat)
    percent_preds = (hist_preds / total_preds) * 100
    percent_coarse = (hist_coarse / total_coarse) * 100

    # Plot bar graphs
    axs[0].bar(bin_edges[:-1], percent_preds, width=bin_widths, alpha=0.6, align='edge')
    axs[1].bar(bin_edges[:-1], percent_coarse, width=bin_widths, alpha=0.6, align='edge')

    # Fix y-axis to 0–100%
    for ax in axs:
        ax.set_ylim(0, 100)
        ax.set_yticks(np.linspace(0, 100, 6))
        ax.set_yticklabels([f"{y:.0f}%" for y in np.linspace(0, 100, 6)])

    # Titles and labels
    axs[0].set_title('FMint-SDE', fontsize=24)
    axs[1].set_title('Coarse Simulation', fontsize=24)
    axs[0].set_xlabel('Error', fontsize=20)
    axs[0].set_ylabel('Percentage', fontsize=20)
    axs[1].set_xlabel('Error', fontsize=20)
    axs[1].set_ylabel('Percentage', fontsize=20)

    # Format ticks and spines
    for ax in axs:
        ax.spines['left'].set_position(('data', 0))
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='both', which='major', labelsize=16)

    fig.tight_layout()

    if to_tfboard:
        return utils.plot_to_image(fig)
    else:
        return fig




	
def icon_ODE_plot_data(equation, caption, data, label, pred, config, to_tfboard = True):
	'''
	plot all figures in demo and prediction
	@param 
		equation: string
		caption: string
	@return
		the figure
	'''
	plt.close('all')
	k_index, qoi_k_Dflag = get_plot_k_index(config['k_mode'], equation)
	if qoi_k_Dflag == 3:
		fig, axs = plt.subplots(3, 1, figsize=(8,6))
		fig.subplots_adjust(hspace=0.4, wspace=0.0)
	elif qoi_k_Dflag == 2:
		fig, axs = plt.subplots(2, 1, figsize=(8, 6))
		fig.subplots_adjust(hspace=0.4, wspace=0.0)
	else:
		fig, axs = plt.subplots(1, 1, figsize=(6, 4))
	
	caption = ""
	# fig.suptitle("eqn:{}\ncaption: {}".format(equation, caption))
	# plot pred for both dimensions 
 
	# print("cond time: ",data.quest_cond_k[0, data.quest_cond_mask[0,:].astype(bool), 0])
	# print("qoi time: ", data.quest_qoi_k[0, data.quest_cond_mask[0,:].astype(bool), 0])

	ind_cond = np.argsort(data.quest_cond_k[0, data.quest_cond_mask[0,:].astype(bool), 0])
	ind_qoi = np.argsort(data.quest_qoi_k[0, data.quest_cond_mask[0,:].astype(bool), 0])

	if qoi_k_Dflag >= 2:
		fine_u_0 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi]
		fine_u_1 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi]

		coarse_u_0 = (pred[data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi]
		coarse_u_1 = (pred[data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi]

		axs[0].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							fine_u_0, 'r+', markersize=15, label='fine-grained ode')
		# axs[0].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
		# 					(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond], 'g+', markersize=15, label='coarse-grained ode')
		axs[0].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							coarse_u_0, '*', markersize=9, label='predicted ode -- icon', alpha = 0.5)

		axs[1].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							fine_u_1, 'r+', markersize=15, label='fine-grained ode')
		# axs[1].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
		# 					(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond], 'g+', markersize=15, label='coarse-grained ode')
		axs[1].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							coarse_u_1, '*', markersize=9, label='predicted ode -- icon', alpha = 0.5)
		if qoi_k_Dflag == 3:
			fine_u_2 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 2])[ind_qoi]
			coarse_u_2 = (pred[data.quest_qoi_mask[0,:].astype(bool), 2])[ind_qoi]

			axs[2].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
								fine_u_2, 'r+', markersize=15, label='fine-grained ode')
			# axs[2].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
			# 					(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond], 'g+', markersize=15, label='coarse-grained ode')
			axs[2].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
								coarse_u_2, '*', markersize=9, label='predicted ode -- icon', alpha = 0.5)
	else:
		fine_u = (label[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi]

		coarse_u = (pred[data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi]

		# print("qoi time: ", (data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]).shape)
		# print("quest cond v shape: ", (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0]).shape)
		# print("label shape: ", (label[0, data.quest_qoi_mask[0,:].astype(bool), 0]).shape)
		# print("fine u: ", fine_u.shape)


		axs.plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							fine_u, 'r+', markersize=15, label='fine-grained ode')
		# axs.plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
		# 					(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond], 'g+', markersize=15, label='coarse-grained ode')
		axs.plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							coarse_u, '*', markersize=9, label='predicted ode -- icon', alpha = 0.5)

	if qoi_k_Dflag >= 2:
		axs[0].set_xlabel('time')
		axs[0].set_ylabel('value')
		axs[0].legend(loc = "upper right", fontsize="25")
		axs[0].yaxis.set_major_locator(ticker.LinearLocator(4))
		axs[0].yaxis.set_minor_locator(ticker.LinearLocator(8))
		axs[1].set_xlabel('time')
		axs[1].set_ylabel('value')
		axs[1].legend(loc = "upper right",fontsize="25")
		axs[1].yaxis.set_major_locator(ticker.LinearLocator(4))
		axs[1].yaxis.set_minor_locator(ticker.LinearLocator(8))

		for item in ([axs[0].xaxis.label, axs[0].yaxis.label] + axs[0].get_xticklabels() + axs[0].get_yticklabels()):
			item.set_fontsize(25)
		for item in ([axs[1].xaxis.label, axs[1].yaxis.label] + axs[1].get_xticklabels() + axs[1].get_yticklabels()):
			item.set_fontsize(25)

		if qoi_k_Dflag == 3:
			axs[2].set_xlabel('time')
			axs[2].set_ylabel('value')
			axs[2].legend(loc = "upper right",fontsize="25")
			axs[2].yaxis.set_major_locator(ticker.LinearLocator(4))
			axs[2].yaxis.set_minor_locator(ticker.LinearLocator(8))

			for item in ([axs[2].xaxis.label, axs[2].yaxis.label] + axs[2].get_xticklabels() + axs[2].get_yticklabels()):
				item.set_fontsize(25)
	else:
		axs.set_xlabel('time'); axs.set_ylabel('value')
		axs.legend(loc = "upper right", fontsize="25")

		axs.yaxis.set_major_locator(ticker.LinearLocator(4))
		axs.yaxis.set_minor_locator(ticker.LinearLocator(8))

		for item in ([axs.xaxis.label, axs.yaxis.label] + axs.get_xticklabels() + axs.get_yticklabels()):
			item.set_fontsize(25)
	
	if to_tfboard:
		return utils.plot_to_image(fig)
	else:  # save to a file
		return fig

	

def plot_keynote(equation, caption, data, label, pred, config, to_tfboard = True):
	'''
	plot all figures in demo and prediction
	@param 
		equation: string
		caption: string
	@return
		the figure
	'''
	plt.close('all')
	k_index, qoi_k_2D_flag = get_plot_k_index(config['k_mode'], equation)
	if qoi_k_2D_flag:
		fig1, axs1 = plt.subplots(2, 1, figsize=(16, 12))
		fig2, axs2 = plt.subplots(2, 1, figsize=(16, 12))
		fig3, axs3 = plt.subplots(2, 1, figsize=(16, 12))
		fig1.patch.set_alpha(0.0)
		fig2.patch.set_alpha(0.0)
		fig3.patch.set_alpha(0.0)
	else:
		fig1, axs1 = plt.subplots(1, 1, figsize=(16, 12))
		fig2, axs2 = plt.subplots(1, 1, figsize=(16, 12))
		fig3, axs3 = plt.subplots(1, 1, figsize=(16, 12))
		fig1.patch.set_alpha(0.0)
		fig2.patch.set_alpha(0.0)
		fig3.patch.set_alpha(0.0)
	fig1.subplots_adjust(hspace=0.2, wspace=0.0)
	caption = ""
	# fig1.suptitle("eqn:{}\ncaption: {}".format(equation, caption))
	# plot cond quest
	
	# cond_quest = prompt[mask_cond_quest, :k_dim+v_dim]  # [cond_len_in_use, k_dim+v_dim]
	ind_cond = np.argsort(data.quest_cond_k[0, data.quest_cond_mask[0,:].astype(bool), 0])
	ind_qoi = np.argsort(data.quest_qoi_k[0, data.quest_cond_mask[0,:].astype(bool), 0])

	if qoi_k_2D_flag:
		fine_u_0 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]
		fine_u_1 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond]

		# coarse_u_0 = (pred[data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]
		# coarse_u_1 = (pred[data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond]

		pred_0 = (pred[data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi]
		pred_1 = (pred[data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi]

		axs1[0].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							fine_u_0, 'o-', color = 'orange', markersize=12, label='fine-grained ode')
		axs1[0].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond], 'go-', markersize=12, label='coarse-grained ode')
		# axs[0].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
		# 					coarse_u_0, '*', markersize=9, label='predicted ode -- icon', alpha = 0.5)

		axs1[1].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							fine_u_1, 'o-', color = 'orange',markersize=12, label='fine-grained ode')
		axs1[1].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond], 'go-', markersize=12, label='coarse-grained ode')
		# axs1[1].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
		# 					coarse_u_1, '*', markersize=9, label='predicted ode -- icon', alpha = 0.5)
  
		axs2[0].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond], 'go-', markersize=15, label='coarse-grained ode')
		axs2[1].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond], 'go-', markersize=15, label='coarse-grained ode')
		
		axs3[0].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(label[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi], 'yo-', markersize=15, label='FG-CG')
		axs3[1].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(label[0, data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi], 'yo-', markersize=15, label='FG-CG')
		
	else:
		fine_u = (label[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi]

		pred = (pred[data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi]

		axs1.plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							fine_u, 'o-', color = 'orange', markersize=12, label='fine-grained ode')
		axs1.plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond], 'go-', markersize=12, label='coarse-grained ode')
		# axs1.plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
		# 					coarse_u, '*', markersize=9, label='predicted ode -- icon', alpha = 0.5)
  
		axs2.plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond], 'go-', markersize=15, label='coarse-grained ode')

		axs3.plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(label[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi], 'yo-', markersize=15, label='FG-CG')

	if qoi_k_2D_flag:
		# axs1[0].set_xlabel('time')
		# axs[0].set_ylabel('value')
		axs1[0].legend(loc = "upper right", fontsize="25")
		axs1[0].set_yticklabels([])
		axs1[0].set_xticklabels([])
		# axs1[0].yaxis.set_major_locator(ticker.LinearLocator(4))
		# axs1[0].yaxis.set_minor_locator(ticker.LinearLocator(8))
		# axs[1].set_xlabel('time')
		# axs[1].set_ylabel('value')
		axs1[1].legend(loc = "upper right",fontsize="25")
		axs1[1].set_yticklabels([])
		axs1[1].set_xticklabels([])
		# axs1[1].yaxis.set_major_locator(ticker.LinearLocator(4))
		# axs1[1].yaxis.set_minor_locator(ticker.LinearLocator(8))

		# for item in ([axs1[0].xaxis.label, axs1[0].yaxis.label] + axs1[0].get_xticklabels() + axs1[0].get_yticklabels()):
		# 	item.set_fontsize(25)
		# for item in ([axs1[1].xaxis.label, axs1[1].yaxis.label] + axs1[1].get_xticklabels() + axs1[1].get_yticklabels()):
		# 	item.set_fontsize(25)
	else:
		# axs.set_xlabel('time'); axs.set_ylabel('value')
		axs1.legend(loc = "upper right", fontsize="25")

		axs1.set_yticklabels([])
		axs1.set_xticklabels([])

		# axs1.yaxis.set_major_locator(ticker.LinearLocator(4))
		# axs1.yaxis.set_minor_locator(ticker.LinearLocator(8))

		# for item in ([axs1.xaxis.label, axs1.yaxis.label] + axs1.get_xticklabels() + axs1.get_yticklabels()):
		# 	item.set_fontsize(25)
	
	return fig1, fig2, fig3
	
	

if __name__ == "__main__":
	pass
