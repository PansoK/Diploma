import sys
import json
import re
import ast
import os
import argparse

import math
import numpy as np
import matplotlib.pyplot as plt

def plot_signature_norms_and_mean(txt_file, attn_type = 'encoder_self', layer_type = 'encoder', plot_every=None, name=None):
	'''
		Only works for an even number of layers and does not check if keys actually exist. 
		Assuming 5 positions to be used for the mean.
	'''
	with open(txt_file, 'r') as f:
		lines = f.readlines()

	# attention
	signature_norms_per_layer = []
	inf_fc_norm_per_layer = []
	proj_per_layer = []
	mean_per_pos_per_layer = []
	mean_attn_before_per_pos_per_layer = []
	mean_attn_after_per_pos_per_layer = []


	# layer
	fc1_per_layer = []

	# counters
	#count = 0
	count_attn = 0
	count_layer = 0
	for line in lines:
		start = int([m.start() for m in re.finditer("{", line)][0])
		stop = int([m.start() for m in re.finditer("}", line)][0]) + 1
		data_dict = ast.literal_eval(line[start: stop])
		# if we have reached layer 0 of the attn_type module then add 1 to count
		if data_dict["name"][0] == attn_type and int(data_dict["name"][1]) == 0:
			count_attn += 1
		if data_dict["name"][0] == layer_type and int(data_dict["name"][1]) == 0:
			count_layer += 1
		# you can check for attention type here
		if attn_type in line and (plot_every is None or count_attn%plot_every == 1):
			# first we only care about sequences that have more than 5 positions
			if (len(data_dict["Multiplying in Mean (Pos: 1-5)"][0]) < 5):
				continue
			# Checking if this layer has been seen before or not
			if int(data_dict["name"][1]) > len(signature_norms_per_layer) - 1:
				signature_norms_per_layer.append([data_dict["Sign Norms"]])
				mean_per_pos_per_layer.append([data_dict["Multiplying in Mean (Pos: 1-5)"]])
				inf_fc_norm_per_layer.append([data_dict["inf_fc1"] + data_dict["inf_fc2"]])
				proj_per_layer.append([data_dict["q_proj"] + data_dict["k_proj"]])
				mean_attn_before_per_pos_per_layer.append([data_dict["Attn Norm Before Multiplication (Pos: 1-5)"]])
				mean_attn_after_per_pos_per_layer.append([data_dict["Attn Norm After Multiplication (Pos: 1-5)"]])
			else:
				signature_norms_per_layer[int(data_dict["name"][1])].append(data_dict["Sign Norms"])
				mean_per_pos_per_layer[int(data_dict["name"][1])].append(data_dict["Multiplying in Mean (Pos: 1-5)"])
				inf_fc_norm_per_layer[int(data_dict["name"][1])].append(data_dict["inf_fc1"] + data_dict["inf_fc2"])
				proj_per_layer[int(data_dict["name"][1])].append(data_dict["q_proj"] + data_dict["k_proj"])
				mean_attn_before_per_pos_per_layer[int(data_dict["name"][1])].append(data_dict["Attn Norm Before Multiplication (Pos: 1-5)"])
				mean_attn_after_per_pos_per_layer[int(data_dict["name"][1])].append(data_dict["Attn Norm After Multiplication (Pos: 1-5)"])
		elif not attn_type in line and (plot_every is None or count_attn%plot_every == 1):
			#norm_dict = json.loads(line[start: stop])
			if int(data_dict["name"][1]) > len(fc1_per_layer) - 1:
				fc1_per_layer.append([data_dict["fc1"]])
			else:
				fc1_per_layer[int(data_dict["name"][1])].append(data_dict["fc1"])


	layers = len(signature_norms_per_layer)

	fig, ax = plt.subplots(int(layers/2), 2, figsize=(8, 8))  # ax of shape (int(layers/2), 2)
	fig.suptitle('{}_{}: Head Signatures'.format(name, attn_type) if name is not None else '{}: Head Signatures'.format(attn_type))
	for layer, signature_norms in enumerate(signature_norms_per_layer):
		axis = ax[layer%int(layers/2)][0 if layer < int(layers/2) else 1]
		axis.title.set_text('Layer {}'.format(layer))
		for head_num in range(len(signature_norms[0])):
			axis.plot(np.arange(len(signature_norms)), np.array([norm[head_num] for norm in signature_norms]))
	plt.savefig("fig_{}_{}_sign.png".format(name, attn_type) if name is not None else 'fig_{}_sign.png'.format(attn_type))
	plt.close(fig)

	for layer, mean_per_pos in enumerate(mean_per_pos_per_layer):
		fig, ax = plt.subplots(2, 3, figsize=(8, 8))  # ax of shape (int(layers/2), 2)
		fig.delaxes(ax[1, 2]) # deleting unused axis
		fig.suptitle('{}_{}: Mean per head & position for layer {}'.format(name, attn_type, layer) \
			if name is not None else '{}: Mean per head & position for layer {}'.format(attn_type, layer))
		mean_per_pos_arr = np.array(mean_per_pos)
		mean_per_pos = np.transpose(mean_per_pos_arr, (2, 1, 0)).tolist()
		for pos, mean_per_head in enumerate(mean_per_pos):
			axis = ax[0 if pos < 3 else 1][pos%3]
			axis.title.set_text('Position {}'.format(pos))
			for head, mean in enumerate(mean_per_head):
				axis.plot(np.arange(len(mean)), np.array(mean))
		handles, labels = axis.get_legend_handles_labels()
		fig.legend(handles, labels, loc='upper center')
		plt.savefig("fig_{}_{}_mean_{}.png".format(name, attn_type, layer) if name is not None else 'fig_{}_mean_{}.png'.format(attn_type, layer))
		plt.close(fig)

	fig, ax = plt.subplots(2, layers, figsize=(12, 8))  # ax of shape (2, layers)
	fig.suptitle('{}_{}: inf_fc1'.format(name, attn_type) if name is not None else '{}: inf_fc1'.format(attn_type))
	for layer, fc_norms in enumerate(inf_fc_norm_per_layer):
		for param in ["weight", "bias"]:
			axis = ax[0 if param == "weight" else 1][layer]
			axis.title.set_text('Layer {}'.format(layer))
			axis.plot(np.arange(len(fc_norms)), np.array([norm[1 if param == "bias" else 0] for norm in fc_norms]))
	plt.savefig("fig_{}_{}_inf_fc1.png".format(name, attn_type) if name is not None else 'fig_{}_inf_fc1.png'.format(attn_type))
	plt.close(fig)

	fig, ax = plt.subplots(2, layers, figsize=(12, 8))  # ax of shape (2, layers)
	fig.suptitle('{}_{}: inf_fc2'.format(name, attn_type) if name is not None else '{}: inf_fc2'.format(attn_type))
	for layer, fc_norms in enumerate(inf_fc_norm_per_layer):
		for param in ["weight", "bias"]:
			axis = ax[0 if param == "weight" else 1][layer]
			axis.title.set_text('Layer {}'.format(layer))
			axis.plot(np.arange(len(fc_norms)), np.array([norm[3 if param == "bias" else 2] for norm in fc_norms]))
	#		print(([norm[3 if param == "bias" else 2] for norm in fc_norms]))
	plt.savefig("fig_{}_{}_inf_fc2.png".format(name, attn_type) if name is not None else 'fig_{}_inf_fc2.png'.format(attn_type))
	plt.close(fig)

	fig, ax = plt.subplots(2, layers, figsize=(12, 8))  # ax of shape (2, layers)
	fig.suptitle('{}_{}: proj'.format(name, attn_type) if name is not None else '{}: proj'.format(attn_type))
	for layer, proj_norms in enumerate(proj_per_layer):
		for param in ["weight", "bias"]:
			axis = ax[0 if param == "weight" else 1][layer]
			axis.title.set_text('Layer {}'.format(layer))
			for proj in ["q", "k"]:
				axis.plot(np.arange(len(proj_norms)), np.array([norm[(0 if proj == "q" else 2) + (1 if param == "bias" else 0)] for norm in proj_norms]), \
					label=proj + '_proj' + '_' + param)
			axis.legend()
	plt.savefig("fig_{}_{}_proj.png".format(name, attn_type) if name is not None else 'fig_{}_proj.png'.format(attn_type))
	plt.close(fig)

	fig, ax = plt.subplots(2, layers, figsize=(12, 8))  # ax of shape (2, layers)
	fig.suptitle('{}_{}: fc1'.format(name, layer_type) if name is not None else '{}: fc1'.format(attn_type))
	for layer, fc1_norms in enumerate(fc1_per_layer):
		for param in ["weight", "bias"]:
			axis = ax[0 if param == "weight" else 1][layer]
			axis.title.set_text('Layer {}'.format(layer))
			axis.plot(np.arange(len(fc1_norms)), np.array([norm[1 if param == "bias" else 0] for norm in fc1_norms]), label=param)
			axis.legend()
	plt.savefig("fig_{}_{}_fc1_norm.png".format(name, layer_type) if name is not None else '{}_fc1_norm.png'.format(attn_type))
	plt.close(fig)

	for layer, mean_per_pos in enumerate(mean_attn_before_per_pos_per_layer):
		fig, ax = plt.subplots(2, 3, figsize=(8, 8))  # ax of shape (int(layers/2), 2)
		fig.delaxes(ax[1, 2]) # deleting unused axis
		fig.suptitle('{}_{}: Mean Attn Before per head & position for layer {}'.format(name, attn_type, layer) \
			if name is not None else '{}: Mean Attn Before per head & position for layer {}'.format(attn_type, layer))
		mean_per_pos_arr = np.array(mean_per_pos)
		mean_per_pos = np.transpose(mean_per_pos_arr, (2, 1, 0)).tolist()
		for pos, mean_per_head in enumerate(mean_per_pos):
			axis = ax[0 if pos < 3 else 1][pos%3]
			axis.title.set_text('Position {}'.format(pos))
			for head, mean in enumerate(mean_per_head):
				axis.plot(np.arange(len(mean)), np.array(mean))
		handles, labels = axis.get_legend_handles_labels()
		fig.legend(handles, labels, loc='upper center')
		plt.savefig("fig_{}_{}_mean_attn_before_{}.png".format(name, attn_type, layer) if name is not None else 'fig_{}_mean_{}.png'.format(attn_type, layer))
		plt.close(fig)

	for layer, mean_per_pos in enumerate(mean_attn_after_per_pos_per_layer):
		fig, ax = plt.subplots(2, 3, figsize=(8, 8))  # ax of shape (int(layers/2), 2)
		fig.delaxes(ax[1, 2]) # deleting unused axis
		fig.suptitle('{}_{}: Mean Attn After per head & position for layer {}'.format(name, attn_type, layer) \
			if name is not None else '{}: Mean Attn After per head & position for layer {}'.format(attn_type, layer))
		mean_per_pos_arr = np.array(mean_per_pos)
		mean_per_pos = np.transpose(mean_per_pos_arr, (2, 1, 0)).tolist()
		for pos, mean_per_head in enumerate(mean_per_pos):
			axis = ax[0 if pos < 3 else 1][pos%3]
			axis.title.set_text('Position {}'.format(pos))
			for head, mean in enumerate(mean_per_head):
				axis.plot(np.arange(len(mean)), np.array(mean))
		handles, labels = axis.get_legend_handles_labels()
		fig.legend(handles, labels, loc='upper center')
		plt.savefig("fig_{}_{}_mean_attn_after_{}.png".format(name, attn_type, layer) if name is not None else 'fig_{}_mean_{}.png'.format(attn_type, layer))
		plt.close(fig)

def plot_norms_for_all(folder, attn_type, layer_type, plot_every):
	'''
		Expected format of txt file: A dictionary per line
		Expected format of txt file name: train_{model_name}_split_{mode}.txt
	'''
	# make list of all files that follow correct type
	files = os.listdir(path=folder)
	# keep only log files
	txt_files = [file.split('.')[0] for file in files if file.split('.')[-1] == 'txt']
	# keep only files of the correct format and call function for them
	for file in txt_files:
		file_parts = file.split('_')
		if file_parts[0] != "train" or file_parts[-2] != "split" or not (file_parts[-1] == 'train' or file_parts[-1] == 'val'):
			continue
		name = '_'.join(file_parts[1:-2] + [file_parts[-1]])
		plot_signature_norms_and_mean(os.path.join(folder, file + '.txt'), attn_type, layer_type, plot_every, name)

def plot_seq_info(txt_file, attn_type = 'encoder_self', layer_type = 'encoder', name=None):
	'''
		The txt file will look like this: The first line will be a dictionary containing the sentences and codes for each sentence
		The next lines will each contain a dictionary with information metrics about one or more sequences. Information about each 
			metric will be stored in a separate dictionary.
	'''
	with open(txt_file, 'r') as f:
		lines = f.readlines()

	start = int([m.start() for m in re.finditer("{", lines[0])][0])
	stop = int([m.start() for m in re.finditer("}", lines[0])][0]) + 1
	sentences = ast.literal_eval(lines[0][start: stop])

	number_of_seqs = len(sentences.keys())
	seq_dict = {}
	for seq_id in sentences.keys():
		seq_dict[seq_id] = {"Multipliers": [], "Norms Before": [], "Norms After": []}
	for line in lines[1:]:
		start = int([m.start() for m in re.finditer("{", line)][0])
		stop = int([m.start() for m in re.finditer("}", line)][-1]) + 1
		data_dict = ast.literal_eval(line[start: stop])
		# Check for correct attn type
		if attn_type in line:
			for seq_id in data_dict["metrics"].keys():
				# check if we have a new unseen layer
				if int(data_dict['name'][1]) > len(seq_dict[seq_id]["Multipliers"]) - 1:
					seq_dict[seq_id]["Multipliers"].append([data_dict["metrics"][seq_id]["Multipliers"]])
					seq_dict[seq_id]["Norms Before"].append([data_dict["metrics"][seq_id]["Norms Before"]])
					seq_dict[seq_id]["Norms After"].append([data_dict["metrics"][seq_id]["Norms After"]])
				else:
					seq_dict[seq_id]["Multipliers"][int(data_dict['name'][1])].append(data_dict["metrics"][seq_id]["Multipliers"])
					seq_dict[seq_id]["Norms Before"][int(data_dict['name'][1])].append(data_dict["metrics"][seq_id]["Norms Before"])
					seq_dict[seq_id]["Norms After"][int(data_dict['name'][1])].append(data_dict["metrics"][seq_id]["Norms After"])

	for seq_id in seq_dict.keys():
		for layer, multiplier_per_pos in enumerate(seq_dict[seq_id]["Multipliers"]):
			fig, ax = plt.subplots(2, 3, figsize=(8, 8))  # ax of shape (int(layers/2), 2)
			fig.suptitle('{}_{}: Seq {}: Multiplier per head & position for layer {}'.format(name, attn_type, seq_id, layer) \
				if name is not None else '{}: Seq {}: Multiplier per head & position for layer {}'.format(attn_type, seq_id, layer))
			multiplier_per_pos_arr = np.array(multiplier_per_pos)
			multiplier_per_pos = np.transpose(multiplier_per_pos_arr, (2, 1, 0)).tolist()
			for pos, multiplier_per_head in enumerate(multiplier_per_pos[:6]):
				axis = ax[0 if pos < 3 else 1][pos%3]
				axis.title.set_text('Word: {}'.format(sentences[seq_id].split()[pos]))
				for head, mult in enumerate(multiplier_per_head):
					axis.plot(np.arange(len(mult)), np.array(mult))
			handles, labels = axis.get_legend_handles_labels()
			fig.legend(handles, labels, loc='upper center')
			plt.savefig("fig_{}_{}_seq_{}_multiplying_{}.png".format(name, attn_type, seq_id, layer) if name is not None else 'fig_{}_seq_{}_multiplying_{}.png'.format(attn_type, seq_id, layer))
			plt.close(fig)
		for layer, norms_per_pos in enumerate(seq_dict[seq_id]["Norms Before"]):
			fig, ax = plt.subplots(2, 3, figsize=(8, 8))  # ax of shape (int(layers/2), 2)
			fig.suptitle('{}_{}: Seq {}: Attention Norms Before per head & position for layer {}'.format(name, attn_type, seq_id, layer) \
				if name is not None else '{}: Seq {}: Attention Norms Before per head & position for layer {}'.format(attn_type, seq_id, layer))
			norms_per_pos_arr = np.array(norms_per_pos)
			norms_per_pos = np.transpose(norms_per_pos_arr, (2, 1, 0)).tolist()
			for pos, norms_per_head in enumerate(norms_per_pos[:6]):
				axis = ax[0 if pos < 3 else 1][pos%3]
				axis.title.set_text('Word: {}'.format(sentences[seq_id].split()[pos]))
				for head, norm in enumerate(norms_per_head):
					axis.plot(np.arange(len(norm)), np.array(norm))
			handles, labels = axis.get_legend_handles_labels()
			fig.legend(handles, labels, loc='upper center')
			plt.savefig("fig_{}_{}_seq_{}_attn_norms_before_{}.png".format(name, attn_type, seq_id, layer) if name is not None else 'fig_{}_seq_{}_attn_norms_before_{}.png'.format(attn_type, seq_id, layer))
			plt.close(fig)
		for layer, norms_per_pos in enumerate(seq_dict[seq_id]["Norms After"]):
			fig, ax = plt.subplots(2, 3, figsize=(8, 8))  # ax of shape (int(layers/2), 2)
			fig.suptitle('{}_{}: Seq {}: Attention Norms After per head & position for layer {}'.format(name, attn_type, seq_id, layer) \
				if name is not None else '{}: Seq {}: Attention Norms After per head & position for layer {}'.format(attn_type, seq_id, layer))
			norms_per_pos_arr = np.array(norms_per_pos)
			norms_per_pos = np.transpose(norms_per_pos_arr, (2, 1, 0)).tolist()
			for pos, norms_per_head in enumerate(norms_per_pos[:6]):
				axis = ax[0 if pos < 3 else 1][pos%3]
				axis.title.set_text('Word: {}'.format(sentences[seq_id].split()[pos]))
				for head, norm in enumerate(norms_per_head):
					axis.plot(np.arange(len(norm)), np.array(norm))
			handles, labels = axis.get_legend_handles_labels()
			fig.legend(handles, labels, loc='upper center')
			plt.savefig("fig_{}_{}_seq_{}_attn_norms_after_{}.png".format(name, attn_type, seq_id, layer) if name is not None else 'fig_{}_seq_{}_attn_norms_after_{}.png'.format(attn_type, seq_id, layer))
			plt.close(fig)

def plot_metrics(log_file, name=None, task='Translation'):
	'''
		Plotting results from a log file in fairseq format. Uses results written at the end of epochs.
	'''


	with open(log_file, 'r') as f:
		lines = f.readlines()
		logs = []
		for line in lines:
			if "| train |" in line or "| valid |" in line or "[train]" in line or "[valid]" in line:
				start = int([m.start() for m in re.finditer("{", line)][0])
				stop = int([m.start() for m in re.finditer("}", line)][0]) + 1
				logs.append(json.loads(line[start: stop]))
	train_losses = [float(log['train_loss']) for log in logs if 'train_loss' in log]
	val_losses = [float(log['valid_loss']) for log in logs if 'valid_loss' in log]

	fig1 = plt.figure(1)
	plt.plot(np.arange(len(train_losses)), np.array(train_losses), label='train')
	plt.plot(np.arange(len(val_losses)), np.array(val_losses), label='val')
	plt.legend();
	plt.title('{}: Losses'.format(name) if name is not None else 'Losses')
	plt.savefig("fig_{}_losses.png".format(name) if name is not None else 'fig_losses.png')

	train_nll_losses = [float(log['train_nll_loss']) for log in logs if 'train_nll_loss' in log]
	val_nll_losses = [float(log['valid_nll_loss']) for log in logs if 'valid_nll_loss' in log]

	fig2 = plt.figure(2)
	plt.plot(np.arange(len(train_nll_losses)), np.array(train_nll_losses), label='train')
	plt.plot(np.arange(len(val_nll_losses)), np.array(val_nll_losses), label='val')
	plt.legend();
	plt.title('{}: NLL Losses'.format(name) if name is not None else 'NLL Losses')
	plt.savefig("fig_{}_nll_losses.png".format(name) if name is not None else 'fig_nll_losses.png')

	train_ppl = [float(log['train_ppl']) for log in logs if 'train_ppl' in log]
	val_ppl = [float(log['valid_ppl']) for log in logs if 'valid_ppl' in log]

	if task == 'Translation':
		val_bleu = [float(log['valid_bleu']) for log in logs if 'valid_bleu' in log]
		fig3, ax1 = plt.subplots()
		ax1.set_yscale('log')
		ax1.set_ylabel('ppl', color='tab:red')
		plt.plot(np.arange(len(train_ppl)), np.array(train_ppl), color='tab:orange', label='train_ppl')
		plt.plot(np.arange(len(val_ppl)), np.array(val_ppl), color='tab:green', label='val_ppl')
		ax1.tick_params(axis='y', labelcolor='tab:red')
		ax1.legend(loc=0);
		ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
		ax2.set_ylabel('BLEU', color='tab:blue')
		plt.plot(np.arange(len(val_bleu)), np.array(val_bleu), color='tab:blue', label='val_bleu')
		ax2.tick_params(axis='y', labelcolor='tab:blue')
		ax2.legend(loc=0);
		plt.title('{}: Perplexities & BLEU'.format(name) if name is not None else 'Perplexities & BLEU')
		fig3.tight_layout()
		plt.savefig("fig_{}_perplexities_bleu.png".format(name) if name is not None else 'fig_perplexities_bleu.png')
	else:
		fig3, ax = plt.subplots()
		ax.set_yscale('log')
		ax.set_ylabel('ppl')
		plt.plot(np.arange(len(train_ppl)), np.array(train_ppl), label='train')
		plt.plot(np.arange(len(val_ppl)), np.array(val_ppl), label='val')
		plt.legend();
		plt.title('{}: Perplexities'.format(name) if name is not None else 'Perplexities')
		plt.savefig("fig_{}_perplexities.png".format(name) if name is not None else 'fig_perplexities.png')


	train_gnorm = [float(log['train_gnorm']) for log in logs if 'train_gnorm' in log]
	train_lr = [float(log['train_lr']) for log in logs if 'train_lr' in log]

	fig4, ax1 = plt.subplots()
	ax1.set_ylabel('Norm', color='tab:red')
	plt.plot(np.arange(len(train_gnorm)), np.array(train_gnorm), color='tab:red', label='train_gnorm')
	ax1.tick_params(axis='y', labelcolor='tab:red')
	ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
	ax2.set_ylabel('lr', color='tab:blue')
	plt.plot(np.arange(len(train_lr)), np.array(train_lr), color='tab:blue', label='train_lr')
	ax2.tick_params(axis='y', labelcolor='tab:blue')
	plt.title('{}: Training Gradient Norm and Learning Rate'.format(name) if name is not None else 'Training Gradient Norm and Learning Rate')
	fig4.tight_layout()
	plt.savefig("fig_{}_train_gnorm_lr.png".format(name) if name is not None else 'fig_train_gnorm_lr.png')

def print_metrics_for_all_seeds(folder):
	'''
		Expected format of log file: Fairseq type
		Expected format of log file name: train_{model_name}_seed{num}.log
	'''
	# make list of all files that follow correct type
	# for loop: make dictionary with name the name of the model and add all data
	files = os.listdir(path=folder)
	# keep only log files
	log_files = [file.split('.')[0] for file in files if file.split('.')[-1] == 'log']
	# keep only files of the correct format and add the model names + seed to a dictionary
	correct_log_files = []
	data = {}
	for file in log_files:
		file_parts = file.split('_')
		if file_parts[0] != "train" or file_parts[-1][:-1] != "seed" or not file_parts[-1][-1].isnumeric():
			continue
		correct_log_files.append(file + '.log')
		data[file] = {}

	for file in correct_log_files:
		with open(folder + file, 'r') as f:
			lines = f.readlines()
			# last valid and train lines must be within the last 10 lines
			valid_line = None
			train_line = None
			for i in range(0, 10):
				if "| train |" in lines[-i]:
					train_line = -i
				elif "| valid |" in lines[-i]:
					valid_line = -i
			if train_line == None or valid_line == None:
				data[file.split('.')[0]]['info'] = "Not correct format"
				continue
			# for train
			start = int([m.start() for m in re.finditer("{", lines[train_line])][0])
			stop = int([m.start() for m in re.finditer("}", lines[train_line])][0]) + 1
			valid_dict = json.loads(lines[train_line][start: stop])
			data[file.split('.')[0]]['train_ppl'] = valid_dict['train_ppl']
			# for valid
			start = int([m.start() for m in re.finditer("{", lines[valid_line])][0])
			stop = int([m.start() for m in re.finditer("}", lines[valid_line])][0]) + 1
			valid_dict = json.loads(lines[valid_line][start: stop])
			data[file.split('.')[0]]['valid_ppl'] = valid_dict['valid_ppl']
			data[file.split('.')[0]]['valid_bleu'] = valid_dict['valid_bleu']
	print(data)		


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
	parser.add_argument('--txt_file', metavar='log_file', required=False, default=None, help='the path to the txt file')
	parser.add_argument('--attn_type', metavar='attn_type', required=False, default='encoder_self', help='attention module results')
	parser.add_argument('--layer_type', metavar='log_file', required=False, default='encoder', help='layer module results')
	parser.add_argument('--plot_every', metavar='plot_every', required=False, default=None, type=int, help='number of iterations between plot points')
	parser.add_argument('--log_file', metavar='log_file', required=False, default=None, help='the path to the log file')
	parser.add_argument('--name', metavar='name', required=False, default=None, help='name of model')
	parser.add_argument('--folder', metavar='path', required=False, default=None, help='folder containing log files')
	parser.add_argument('--task', metavar='task', required=False, default='Translation', help='task')
	args = parser.parse_args()
	#plot_signature_norms_and_mean(args.txt_file, args.attn_type, args.layer_type, args.plot_every, args.name)
	#plot_norms_for_all(args.folder, args.attn_type, args.layer_type, args.plot_every)
	plot_seq_info(args.txt_file, args.attn_type, args.layer_type, args.name)
	#plot_metrics(args.log_file, args.name, args.task)
	#print_metrics_for_all_seeds(args.folder)
