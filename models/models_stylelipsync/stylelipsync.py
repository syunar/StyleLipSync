# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
from utils import common
from models.models_common import BaseModel
from models.models_common.layers import normalize_2nd_moment, modulated_conv2d
from models.models_common.layers import FullyConnectedLayer, Conv2dLayer, ResidualBlock

from models.models_stylelipsync.encoders import AudioEncoder, FaceEncoder, LipEncoder

#----------------------------------------------------------------------------
class SynthesisLayer(torch.nn.Module):
	def __init__(self,
		in_channels,                    # Number of input channels.
		out_channels,                   # Number of output channels.
		w_dim,                          # Intermediate latent (W) dimensionality.
		resolution,                     # Resolution of this layer.
		kernel_size     = 3,            # Convolution kernel size.
		up              = 1,            # Integer upsampling factor.
		use_noise       = True,         # Enable noise input?
		activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
		resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
		conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
		channels_last   = False,        # Use channels_last format for the weights?
	):
		super().__init__()
		self.resolution = resolution
		self.up = up
		self.use_noise = use_noise
		self.activation = activation
		self.conv_clamp = conv_clamp
		self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
		self.padding = kernel_size // 2
		self.act_gain = bias_act.activation_funcs[activation].def_gain

		self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
		self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
		if use_noise:
			self.register_buffer('noise_const', torch.randn([resolution, resolution]))
			self.noise_strength = torch.nn.Parameter(torch.zeros([]))
		self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

	def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
		assert noise_mode in ['random', 'const', 'none']
		in_resolution = self.resolution // self.up
		styles = self.affine(w)

		noise = None
		if self.use_noise and noise_mode == 'random':
			noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
		if self.use_noise and noise_mode == 'const':
			noise = self.noise_const * self.noise_strength

		flip_weight = (self.up == 1) # slightly faster
		x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
			padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

		act_gain = self.act_gain * gain
		act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
		x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
		return x


#----------------------------------------------------------------------------
class ToRGBLayer(torch.nn.Module): # DONE
	def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
		super().__init__()
		self.conv_clamp = conv_clamp
		self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
		self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
		self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
		self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

	def forward(self, x, w, fused_modconv=True):
		styles = self.affine(w) * self.weight_gain
		x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
		x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
		return x

#----------------------------------------------------------------------------

class SynthesisBlock(torch.nn.Module):
	def __init__(self,
		in_channels,                        # Number of input channels, 0 = first block.
		out_channels,                       # Number of output channels.
		w_dim,                              # Intermediate latent (W) dimensionality.
		resolution,                         # Resolution of this block.
		img_channels,                       # Number of output color channels.
		is_last,                            # Is this the last block?
		architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
		resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
		conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
		use_fp16            = False,        # Use FP16 for this block?
		fp16_channels_last  = False,        # Use channels-last memory format with FP16?
		**layer_kwargs,                     # Arguments for SynthesisLayer.
	):
		assert architecture in ['orig', 'skip', 'resnet']
		super().__init__()
		self.in_channels = in_channels
		self.w_dim = w_dim
		self.resolution = resolution
		self.img_channels = img_channels
		self.is_last = is_last
		self.architecture = architecture
		self.use_fp16 = use_fp16
		self.channels_last = (use_fp16 and fp16_channels_last)
		self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
		self.num_conv = 0
		self.num_torgb = 0

		if in_channels == 0:
			self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

		if in_channels != 0:
			self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
				resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
			self.num_conv += 1

		self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
			conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
		self.num_conv += 1

		if is_last or architecture == 'skip':
			self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
				conv_clamp=conv_clamp, channels_last=self.channels_last)
			self.num_torgb += 1

		if in_channels != 0 and architecture == 'resnet':
			self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
				resample_filter=resample_filter, channels_last=self.channels_last)

	def forward(self, x, img, ws, short_cut_, mixes_, force_fp32=False, fused_modconv=None, **layer_kwargs):
		w_iter = iter(ws.unbind(dim=1))
		# print(list(w_iter).__len__())
		dtype = torch.float32
		if fused_modconv is None:
			fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

		# Input.
		if self.in_channels == 0:
			x = self.const.to(dtype=dtype)
			x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
		else:
			x = x.to(dtype=dtype)

		# Main layers.
		if self.in_channels == 0:
			x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
		elif self.architecture == 'resnet':
			y = self.skip(x, gain=np.sqrt(0.5))
			x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
			x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
			x = y.add_(x)
		else:
			w_cur_ = next(w_iter)
			if short_cut_ is not None and mixes_ is not None:
				sigma = mixes_(short_cut_, w_cur_)
				x = sigma * short_cut_ + (1 - sigma) * x
			x = self.conv0(x, w_cur_, fused_modconv=fused_modconv, **layer_kwargs)
			x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
		# ToRGB.
		if img is not None:
			img = upfirdn2d.upsample2d(img, self.resample_filter)
		if self.is_last or self.architecture == 'skip':
			y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
			img = img.add_(y) if img is not None else y

		assert x.dtype == dtype
		assert img is None or img.dtype == torch.float32
		return x, img

#----------------------------------------------------------------------------

class SynthesisNetwork(torch.nn.Module):
	def __init__(self,
		w_dim,                      # Intermediate latent (W) dimensionality.
		img_resolution,             # Output image resolution.
		img_channels,               # Number of color channels.
		channel_base    = 32768,    # Overall multiplier for the number of channels.
		channel_max     = 512,      # Maximum number of channels in any layer.
		num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
		**block_kwargs,             # Arguments for SynthesisBlock.
	):
		assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
		super().__init__()
		self.w_dim = w_dim
		self.img_resolution = img_resolution
		self.img_resolution_log2 = int(np.log2(img_resolution))
		self.img_channels = img_channels
		self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
		channels_dict = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 64}
		self.channels_dict = channels_dict

		self.num_ws = 0
		for res in self.block_resolutions:
			in_channels = channels_dict[res // 2] if res > 4 else 0
			out_channels = channels_dict[res]
			is_last = (res == self.img_resolution)
			block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res, architecture='skip',
				img_channels=img_channels, is_last=is_last, **block_kwargs)
			self.num_ws += block.num_conv
			if is_last:
				self.num_ws += block.num_torgb
			setattr(self, f'b{res}', block)

	def forward(self, ws, short_cut=None, mixes=None, **block_kwargs):
		block_ws = []
		with torch.autograd.profiler.record_function('split_ws'):
			ws = ws.to(torch.float32)
			w_idx = 0
			for res in self.block_resolutions:
				block = getattr(self, f'b{res}')
				block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
				w_idx += block.num_conv

		short_cut = [None] + short_cut # [None] for 4x4
		mixes = [None] + mixes
		x = img = None
		for res, cur_ws, short_cut_, mixes_ in zip(self.block_resolutions, block_ws, short_cut, mixes):
			block = getattr(self, f'b{res}')
			x, img = block(x, img, cur_ws, short_cut_, mixes_, **block_kwargs)
		return img


#----------------------------------------------------------------------------
class StyleLipSync(BaseModel):
	def __init__(self, opt):
		super().__init__()
		self.opt = opt
		self.w_dim = opt.dim_w
		self.a_dim = opt.dim_a
		self.l_dim = opt.dim_l

		self.img_channels = opt.input_nc
		self.img_resolution = opt.input_size

		self.synthesis = SynthesisNetwork(
				w_dim=self.w_dim,
				img_resolution=self.img_resolution,
				img_channels=self.img_channels)

		self.num_ws = self.synthesis.num_ws
		self.audio_encoder = AudioEncoder(opt=opt)
		self.face_encoder = FaceEncoder(opt=opt, channels_dict=self.synthesis.channels_dict)
		self.lip_encoder = LipEncoder(opt=opt, channels_dict=self.synthesis.channels_dict)

		self.mals = nn.ModuleList([MaLS(opt=opt) for _ in range(self.num_ws)])

		self.w_avg = None
		self.load_w_avg(self.opt.w_avg_path, self.num_ws)

	def load_w_avg(self, w_avg_path, num_ws):
		assert num_ws is not None
		if os.path.exists(w_avg_path):
			self.w_avg = torch.load(w_avg_path)['w_avg']
			print("- average latent w loaded: {}".format(w_avg_path))
		else:
			print("- average latent w not loaded")

	def expand_ws(self, x):
		if self.num_ws is not None:
			with torch.autograd.profiler.record_function('broadcast'):
				x = x.unsqueeze(1).repeat([1, self.num_ws, 1])
		return x

	def set_requires_grad(self, requires_grad):
		if not self.opt.adaptation:
			for name, param in self.named_parameters():
				if 'synthesis' in name:
					param.requires_grad_(False)
				else:
					param.requires_grad_(requires_grad)
		else:
			for name, param in self.named_parameters():
				if 'synthesis' in name:
					param.requires_grad_(requires_grad)
				else:
					param.requires_grad_(False)

	def get_parameters_for_train(self):
		if not self.opt.adaptation:
			params = list(self.audio_encoder.parameters()) \
					+ list(self.face_encoder.parameters()) \
					+ list(self.lip_encoder.parameters()) \
					+ list(self.mals.parameters())
		else:
			params = list(self.synthesis.parameters())
		return params

	def merge_embeddings(self, ws_a, ws_l):
		return ws_a + ws_l # boardcast

	def forward(self, data, update_emas=False):
		x, a, l = data['x'], data['mel'], data['ref']
		B, T = x.shape[:2]

		_, shortcuts, mixes = self.face_encoder(x)
		ws_a     		    = self.audio_encoder(a)
		ws_l     		    = self.lip_encoder(l)

		ws_a = ws_a.reshape(B, T, *ws_a.shape[1:])
		ws_l = ws_l.reshape(B, 1, *ws_l.shape[1:])
		ws_l = ws_l.repeat(1, T, 1, 1)

		ws = self.merge_embeddings(ws_a, ws_l)
		ws_smooth = [] 
		for i in range(self.num_ws):
			ws_i = self.mals[i](ws[:, :, i])
			ws_smooth.append(ws_i)
		ws_smooth = torch.stack(ws_smooth, dim=2)
		ws_smooth = ws_smooth + ws_l
		ws_smooth = ws_smooth.reshape(B*T, *ws_smooth.shape[2:])

		if self.w_avg is not None:
			ws_smooth = ws_smooth + self.w_avg.to(ws_smooth.device)

		shortcuts = shortcuts[::-1]
		mixes = mixes[::-1]
		img = self.synthesis(ws_smooth, short_cut=shortcuts, mixes=mixes)
		img = img.reshape(B, T, *img.shape[1:])
		return {'x_hat': img}


	def inference(self, data):
		x, a, l = data['x'], data['mel'], data['ref']
		B, T = x.shape[:2]
		ws_f, shortcuts, ws_a = [], [], []
		mixes = []
		l = l.to(self.opt.rank)
		ws_l = self.lip_encoder(l).unsqueeze(0)
		for t in range(0, T, self.opt.batch_size):
			xt = x[:, t:t+self.opt.batch_size].to(self.opt.rank)
			at = a[:, t:t+self.opt.batch_size].to(self.opt.rank)
			_, shortcuts_t, mixes_t = self.face_encoder(xt)
			ws_at = self.audio_encoder(at)
			
			ws_a.append(ws_at)
			shortcuts.append(shortcuts_t)
			mixes.append(mixes_t)
		ws_a = torch.cat(ws_a, dim=0).unsqueeze(0)
		ws = self.merge_embeddings(ws_a, ws_l)

		ws_smooth = []
		for i in range(self.num_ws):
			ws_i = self.mals[i](ws[:,:,i])
			ws_smooth.append(ws_i)
		ws_smooth = torch.stack(ws_smooth, dim=2) + ws_l

		ws_smooth = ws_smooth.reshape(B*T, *ws_smooth.shape[2:])
		if self.w_avg is not None:
			ws_smooth = ws_smooth + self.w_avg.to(ws_smooth.device)

		imgs = []
		for i in range(len(shortcuts)):
			shortcut_t = shortcuts[i][::-1]
			mixes_t = mixes[i][::-1]
			ws_smooth_t = ws_smooth[self.opt.batch_size * i: self.opt.batch_size * (i+1)]
			img_t = self.synthesis(ws_smooth_t, short_cut=shortcut_t, mixes=mixes_t, noise_mode='const')
			imgs.append(img_t)
		x_hat = torch.cat(imgs, dim=0)
		return {'x_hat': x_hat}


class MaLS(nn.Module):
	"""
		Moving-average based Latent Smoothing Module
	"""
	def __init__(self, opt):
		super().__init__()
		self.opt = opt
		self.conv = nn.Sequential(
			nn.Conv1d(self.opt.dim_w, self.opt.dim_w, 3, 1, 1, padding_mode="reflect"),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv1d(self.opt.dim_w, self.opt.dim_w, 3, 1, 1, padding_mode="reflect"),
		)

	def moving_average(self, ws):
		# ws: T x 512
		ws_start = ws[:, 0:1]
		ws_end   = ws[:, -1:]
		ws = torch.cat([ws_start, ws, ws_end], dim=1)
		ws_p = 0.5 * ws[:, 2:] + 1 * ws[:, 1:-1] + 0.5 * ws[:, :-2]
		ws_p = ws_p / 2
		return ws_p

	def forward(self, ws):
		ws = self.moving_average(ws)
		ws = self.conv(ws.permute(0, 2, 1)).permute(0, 2, 1)
		return ws
