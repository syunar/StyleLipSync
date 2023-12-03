import math
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma


def normalize_2nd_moment(x, dim=-1, eps=1e-8):
    	return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

def modulated_conv2d(
	x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
	weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
	styles,                     # Modulation coefficients of shape [batch_size, in_channels].
	noise           = None,     # Optional noise tensor to add to the output activations.
	up              = 1,        # Integer upsampling factor.
	down            = 1,        # Integer downsampling factor.
	padding         = 0,        # Padding with respect to the upsampled image.
	resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
	demodulate      = True,     # Apply weight demodulation?
	flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
	fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
	batch_size = x.shape[0]
	out_channels, in_channels, kh, kw = weight.shape

	# Pre-normalize inputs to avoid FP16 overflow.
	if demodulate:
		weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
		styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I
	# Calculate per-sample weights and demodulation coefficients.
	w = None
	dcoefs = None
	if demodulate or fused_modconv:
		w = weight.unsqueeze(0) # [NOIkk]
		w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
	if demodulate:
		dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
	if demodulate and fused_modconv:
		w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

	# Execute by scaling the activations before and after the convolution.
	if not fused_modconv:
		x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
		x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
		if demodulate and noise is not None:
			x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
		elif demodulate:
			x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
		elif noise is not None:
			x = x.add_(noise.to(x.dtype))
		return x

	# Execute as one fused op using grouped convolution.
	batch_size = int(batch_size)
	x = x.reshape(1, -1, *x.shape[2:])
	w = w.reshape(-1, in_channels, kh, kw)
	x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
	x = x.reshape(batch_size, -1, *x.shape[2:])
	if noise is not None:
		x = x.add_(noise)
	return x


class FullyConnectedLayer(torch.nn.Module):
	def __init__(self,
		in_features,                # Number of input features.
		out_features,               # Number of output features.
		bias            = True,     # Apply additive bias before the activation function?
		activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
		lr_multiplier   = 1,        # Learning rate multiplier.
		bias_init       = 0,        # Initial value for the additive bias.
	):
		super().__init__()
		self.activation = activation
		self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
		self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
		self.weight_gain = lr_multiplier / np.sqrt(in_features)
		self.bias_gain = lr_multiplier

	def forward(self, x):
		w = self.weight.to(x.dtype) * self.weight_gain
		b = self.bias
		if b is not None:
			b = b.to(x.dtype)
			if self.bias_gain != 1:
				b = b * self.bias_gain

		if self.activation == 'linear' and b is not None:
			x = torch.addmm(b.unsqueeze(0), x, w.t())
		else:
			x = x.matmul(w.t())
			x = bias_act.bias_act(x, b, act=self.activation)
		return x


class Conv2dLayer(torch.nn.Module):
	def __init__(self,
		in_channels,                    # Number of input channels.
		out_channels,                   # Number of output channels.
		kernel_size,                    # Width and height of the convolution kernel.
		bias            = True,         # Apply additive bias before the activation function?
		activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
		up              = 1,            # Integer upsampling factor.
		down            = 1,            # Integer downsampling factor.
		resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
		conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
		channels_last   = False,        # Expect the input to have memory_format=channels_last?
		trainable       = True,         # Update the weights of this layer during training?
	):
		super().__init__()
		self.activation = activation
		self.up = up
		self.down = down
		self.conv_clamp = conv_clamp
		self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
		self.padding = kernel_size // 2
		self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
		self.act_gain = bias_act.activation_funcs[activation].def_gain

		memory_format = torch.channels_last if channels_last else torch.contiguous_format
		weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
		bias = torch.zeros([out_channels]) if bias else None
		if trainable:
			self.weight = torch.nn.Parameter(weight)
			self.bias = torch.nn.Parameter(bias) if bias is not None else None
		else:
			self.register_buffer('weight', weight)
			if bias is not None:
				self.register_buffer('bias', bias)
			else:
				self.bias = None

	def forward(self, x, gain=1):
		w = self.weight * self.weight_gain
		b = self.bias.to(x.dtype) if self.bias is not None else None
		flip_weight = (self.up == 1) # slightly faster
		x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

		act_gain = self.act_gain * gain
		act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
		x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
		return x


class ResidualBlock(torch.nn.Module):
	def __init__(self,
		in_channels,                        # Number of input channels, 0 = first block.
		out_channels,                       # Number of output channels.
		activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
		resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
		conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
		trainable           = True
	):
		super().__init__()
		self.in_channels = in_channels
		self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

		self.num_layers = 0
		self.conv0 = Conv2dLayer(in_channels, out_channels, kernel_size=3, activation=activation,
			trainable=trainable, conv_clamp=conv_clamp,)
		self.num_layers +=1
		self.conv1 = Conv2dLayer(out_channels, out_channels, kernel_size=3, activation=activation, down=2,
			trainable=trainable, resample_filter=resample_filter, conv_clamp=conv_clamp,)
		self.num_layers +=1
		self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, down=2,
			trainable=trainable, resample_filter=resample_filter,)

	def forward(self, x):
		# Main layers.
		y = self.skip(x, gain=np.sqrt(0.5))
		x = self.conv0(x)
		x = self.conv1(x, gain=np.sqrt(0.5))
		x = y.add_(x)
		return x

class ResidualBlock(torch.nn.Module):
	def __init__(self,
		in_channels,                        # Number of input channels, 0 = first block.
		out_channels,                       # Number of output channels.
	):
		super().__init__()
		self.in_channels = in_channels
		
		self.conv0 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
		self.bn0   = nn.BatchNorm2d(out_channels) 

		self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
		self.bn1   = nn.BatchNorm2d(out_channels)
		self.skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

	def forward(self, x):
		# Main layers.
		skip = self.skip(x)

		x = self.conv0(x)
		x = self.bn0(x)
		x = F.relu(x)

		x = self.conv1(x)
		x = self.bn1(x)
		x = x + skip
		x = F.relu(x)

		return x


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out
