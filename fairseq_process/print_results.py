import sys
import json
import re
import ast

import math
import numpy as np
import matplotlib.pyplot as plt

def plot_signature_norms_and_mean(txt_file, attn_type = 'encoder_self', plot_every=200):
	'''
		Only works for an even number of layers. Assuming 5 positions to be used for the mean
	'''
	with open(txt_file, 'r') as f:
		lines = f.readlines()
	signature_norms_per_layer = []
	mean_per_pos_per_layer = []
	count = 0
	for line in lines:
		start = int([m.start() for m in re.finditer("{", line)][0])
		stop = int([m.start() for m in re.finditer("}", line)][0]) + 1
		attn_dict = ast.literal_eval(line[start: stop])
		# if we have reached layer 0 of the attn_type module then add 1 to count
		if attn_dict["name"][0] == attn_type and int(attn_dict["name"][1]) == 0:
			count += 1
		# you can check for attention type here
		if attn_type in line and count%plot_every == 1:
			#norm_dict = json.loads(line[start: stop])
			if int(attn_dict["name"][1]) > len(signature_norms_per_layer) - 1:
				signature_norms_per_layer.append([attn_dict["Sign Norms"]])
				mean_per_pos_per_layer.append([attn_dict["Multplying in Mean (Pos: 1-5)"]])
			else:
				signature_norms_per_layer[int(attn_dict["name"][1])].append(attn_dict["Sign Norms"])
				mean_per_pos_per_layer[int(attn_dict["name"][1])].append(attn_dict["Multplying in Mean (Pos: 1-5)"])
	layers = len(signature_norms_per_layer)

	fig, ax = plt.subplots(int(layers/2), 2, figsize=(8, 8))  # ax of shape (int(layers/2), 2)
	fig.suptitle('Head Signatures')
	for layer, signature_norms in enumerate(signature_norms_per_layer):
		axis = ax[layer%int(layers/2)][0 if layer < int(layers/2) else 1]
		axis.title.set_text('Layer {}'.format(layer))
		for head_num in range(len(signature_norms[0])):
			axis.plot(np.arange(len(signature_norms)), np.array([norm[head_num] for norm in signature_norms]))
	plt.savefig("fig_sign.png")

	for layer, mean_per_pos in enumerate(mean_per_pos_per_layer):
		fig, ax = plt.subplots(2, 3, figsize=(8, 8))  # ax of shape (int(layers/2), 2)
		fig.delaxes(ax[1, 2]) # deleting unused axis
		fig.suptitle('Mean per head & position for layer {}'.format(layer))
		mean_per_pos_arr = np.array(mean_per_pos)
		mean_per_pos = np.transpose(mean_per_pos_arr, (2, 1, 0)).tolist()
		for pos, mean_per_head in enumerate(mean_per_pos):
			axis = ax[0 if pos < 3 else 1][pos%3]
			axis.title.set_text('Position {}'.format(pos))
			for head, mean in enumerate(mean_per_head):
				axis.plot(np.arange(len(mean)), np.array(mean))
		handles, labels = axis.get_legend_handles_labels()
		fig.legend(handles, labels, loc='upper center')
		plt.savefig("fig_mean_{}.png".format(layer))

def plot_metrics(log_file):
	with open(log_file, 'r') as f:
		lines = f.readlines()
		logs = []
		for line in lines:
			if "| train |" in line or "| valid |" in line:
				start = int([m.start() for m in re.finditer("{", line)][0])
				stop = int([m.start() for m in re.finditer("}", line)][0]) + 1
				logs.append(json.loads(line[start: stop]))
	train_losses = [float(log['train_loss']) for log in logs if 'train_loss' in log]
	val_losses = [float(log['valid_loss']) for log in logs if 'valid_loss' in log]

	fig1 = plt.figure(1)
	plt.plot(np.arange(len(train_losses)), np.array(train_losses), label='train')
	plt.plot(np.arange(len(val_losses)), np.array(val_losses), label='val')
	plt.legend();
	plt.title('Losses')
	plt.savefig("losses.png")

	train_nll_losses = [float(log['train_nll_loss']) for log in logs if 'train_nll_loss' in log]
	val_nll_losses = [float(log['valid_nll_loss']) for log in logs if 'valid_nll_loss' in log]

	fig2 = plt.figure(2)
	plt.plot(np.arange(len(train_nll_losses)), np.array(train_nll_losses), label='train')
	plt.plot(np.arange(len(val_nll_losses)), np.array(val_nll_losses), label='val')
	plt.legend();
	plt.title('NLL Losses')
	plt.savefig("nll_losses.png")

	train_ppl = [float(log['train_ppl']) for log in logs if 'train_ppl' in log]
	val_ppl = [float(log['valid_ppl']) for log in logs if 'valid_ppl' in log]
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
	plt.title('Perplexities & BLEU')
	fig3.tight_layout()
	plt.savefig("perplexities_bleu.png")

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
	plt.title('Training Gradient Norm and Learning Rate')
	fig4.tight_layout()
	plt.savefig("train_gnorm_lr.png")


if __name__ == '__main__':
    #plot_signature_norms_and_mean(sys.argv[1])
    plot_metrics(sys.argv[1])