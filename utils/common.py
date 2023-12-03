import numpy as np
import os, cv2, subprocess
import torch, torchaudio
import matplotlib
import torch.nn.functional as F

from PIL import Image

def _3dto2d(tensor):
	return tensor.reshape(tensor.shape[0]*tensor.shape[1], *tensor.shape[2:]) if len(tensor.shape) == 5 else tensor

def tensor2cuda(tensor, rank):
	if type(tensor) is list or type(tensor) is tuple:
		return [t.to(rank) for t in tensor]
	elif type(tensor) == dict:
		return {k: v.to(rank) for k, v in tensor.items()}
	else:
		return tensor.to(rank)

def aggregate_loss_dict(agg_loss_dict):
	mean_vals = {}
	for output in agg_loss_dict:
		for key in output:
			mean_vals[key] = mean_vals.setdefault(key, []) + [output[key]]
	for key in mean_vals:
		if len(mean_vals[key]) > 0:
			mean_vals[key] = sum(mean_vals[key]) / len(mean_vals[key])
		else:
			print('{} has no value'.format(key))
			mean_vals[key] = 0
	return mean_vals

def batch2im(var):
	var = var.cpu().detach().permute((0,2,3,1)).numpy()
	var = (var+1)/2
	var[var<0]=0
	var[var>1]=1
	var=var*255
	return var.astype('uint8')

def tensor2im(var, drange=[-1, 1], mode='RGB'):
	lo, hi = drange
	var = var.cpu().detach().permute((1, 2, 0)).numpy()
	var = (var - lo) * (255 / (hi - lo))
	var = np.rint(var).clip(0, 255).astype(np.uint8)
	var = var if var.shape[-1] != 1 else var[..., 0] # for gray scale image
	return Image.fromarray(var)
