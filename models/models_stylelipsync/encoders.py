import math
import torch
import numpy as np

from torch import nn
import torch.nn.functional as F
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma

from models.models_common import BaseModel
from models.models_common.resnet_se import se_resnet18
from models.models_common.layers import normalize_2nd_moment, modulated_conv2d
from models.models_common.layers import FullyConnectedLayer, Conv2dLayer


## Audio Encoder
class AudioEncoder(nn.Module):
	def __init__(self, opt):
		super().__init__()

		self.dim_w = opt.dim_w
		self.input_size = opt.input_size
		self.audio_encoder = se_resnet18()

		self.audio_encoder.stem = nn.Sequential(
			nn.Conv2d(1, 64, 3, 1, 1, bias=False),
			nn.BatchNorm2d(64),
			nn.PReLU(64),
			nn.MaxPool2d(kernel_size=3, stride=(2,2), padding=1)
		)
	
		self.audio_encoder.fc = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(512, 2*(int(np.log2(self.input_size)) - 1) * self.dim_w)
		)
			
	def load_weight(self, path):
		state_dict = torch.load(path, map_location='cpu')
		self.load_state_dict(state_dict)
		del state_dict

	def forward(self, x):
		B, T = x.shape[:2]
		x = x.reshape(B*T, *x.shape[2:])
		# x : B x 20 x 80
		x = self.audio_encoder(x.unsqueeze(1)) # B x self.proj_dim
		x = x.reshape(-1, 2*(int(np.log2(self.input_size)) - 1), self.dim_w)
		return x

class FaceEncoder(nn.Module):
	def __init__(self, opt, channels_dict):
		super().__init__()		
		self.dim_w = opt.dim_w
		self.img_channels = opt.input_nc
		self.img_resolution = opt.input_size
		self.channels_dict = channels_dict
		self.log_size = int(math.log(self.img_resolution, 2))

		in_channel = channels_dict[self.img_resolution]
		self.f_enc0 = nn.Sequential(*[SameBlock2d(self.img_channels,
			self.channels_dict[self.img_resolution], kernel_size=(7, 7), padding=(3, 3))])
		self.names = ["f_enc%d"%i for i in range(self.log_size-1)]
		self.names_mix = ["f_mix%d"%i for i in range(self.log_size-1)]

		for i in range(self.log_size, 2, -1):
			out_channel = channels_dict[2**(i-1)]
			conv = [DownBlock2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1))]
			mix  = ResidualAttention(out_channel, 1, self.dim_w)
			
			setattr(self, self.names[self.log_size-i+1], nn.Sequential(*conv))
			setattr(self, self.names_mix[self.log_size-i+1], mix)
			in_channel = out_channel
		
	def forward(self, x):
		if len(x.shape) == 5:
			B, T, C, H, W = x.shape
			x = x.reshape(B*T, *x.shape[2:])
		features = []
		mixes = []
		for i in range(self.log_size - 1):
			ecd = getattr(self, self.names[i])
			x = ecd(x)
			if i > 0:
				features.append(x)
				mix = getattr(self, self.names_mix[i])
				mixes.append(mix)
		# x = self.fcs(x).squeeze().squeeze() # B*T x 512
		return None, features, mixes

class ResidualAttention(torch.nn.Module):
	def __init__(self,
		in_channels,                    # Number of input channels.
		out_channels,                   # Number of output channels.
		w_dim,                          # Intermediate latent (W) dimensionality.
		kernel_size     = 3,            # Convolution kernel size.
		activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
		resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
		conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
		channels_last   = False,        # Use channels_last format for the weights?
	):
		super().__init__()
		self.activation = activation
		self.conv_clamp = conv_clamp
		self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
		self.padding = kernel_size // 2
		self.act_gain = bias_act.activation_funcs[activation].def_gain

		self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
		self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
		self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

	def forward(self, x, w, fused_modconv=True, gain=1):
		styles = self.affine(w)
		x = modulated_conv2d(x=x, weight=self.weight, styles=styles, padding=self.padding,
				resample_filter=self.resample_filter, flip_weight=True, fused_modconv=fused_modconv)
		act_gain = self.act_gain * gain
		act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
		x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
		x = torch.sigmoid(x)
		return x


class LipEncoder(nn.Module):
	def __init__(self, opt, channels_dict):
		super().__init__()
		self.dim_l = opt.dim_l
		self.dim_w = opt.dim_w
		self.img_channels = opt.input_nc
		self.img_resolution = opt.input_size
		self.channels_dict = channels_dict
		self.log_size = int(math.log(self.img_resolution, 2)) 

		in_channel = channels_dict[self.img_resolution]
		self.l_enc0 = nn.Sequential(*[SameBlock2d(self.img_channels,
			self.channels_dict[self.img_resolution], kernel_size=(7, 7), padding=(3, 3))])
		self.names = ["l_enc%d"%i for i in range(self.log_size-1)]

		for i in range(self.log_size, 2, -1):
			out_channel = channels_dict[2**(i-1)]
			conv = [DownResidualBlock2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1))]
			setattr(self, self.names[self.log_size-i+1], nn.Sequential(*conv))
			in_channel = out_channel

		self.fcs = nn.Sequential(
			nn.Flatten(),
			nn.Linear(4 * 4 * 512, self.dim_w * 2 * (self.log_size - 1))
			)		

	def forward(self, x):
		if len(x.shape) == 5:
			B, T, C, H, W = x.shape
			x = x.reshape(B*T, *x.shape[2:])
		for i in range(self.log_size -1):
			enc_block = getattr(self, self.names[i])
			x = enc_block(x)
		return self.fcs(x).reshape(-1, 2 * (self.log_size - 1), self.dim_w)
		
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

class DownResidualBlock2d(nn.Module):
	def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
		super().__init__()
		self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
							padding=padding, groups=groups)
		self.norm = nn.InstanceNorm2d(out_features, affine=True)
		self.pool = nn.AvgPool2d(kernel_size=(2, 2))

		self.residual = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1,
							stride=2, padding=0)

	def forward(self, x):
		identity = x
		out = self.conv(x)
		out = self.norm(out)
		out = F.relu(out)
		out = self.pool(out)
		out = out + self.residual(identity)
		return F.relu(out)
