import os, math, shutil, pickle, copy, yaml, random, json, cv2
import torch, torchvision, torchaudio
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import albumentations as A
import albumentations.pytorch.transforms as A_pytorch

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from utils import common

class BaseDataLoader:
	def __init__(self, opt):
		self.opt = opt
		self.input_size = opt.input_size
		self.input_nc   = opt.input_nc
		self.image_size = (opt.input_size, opt.input_size)
		self.num_frames_per_clip = opt.num_frames_per_clip
		
		self.fps = opt.fps
		self.bps = opt.sampling_rate / opt.hop_length
		self.sampling_rate = opt.sampling_rate
		self.num_mel_bins  = int(self.bps * self.num_frames_per_clip / self.fps)
		
		self.img_transform = A.Compose([
			A.Resize(height= self.input_size, width = self.input_size, interpolation=cv2.INTER_AREA),
			A.Normalize(mean=(0.5, 0.5, 0.5), std= (0.5, 0.5, 0.5)),
			A_pytorch.ToTensorV2(),
		])
		self.audio_transform = torchaudio.transforms.MelSpectrogram(
			sample_rate=opt.sampling_rate, n_mels=opt.n_mels,
			n_fft=opt.n_fft, win_length=opt.win_length, hop_length=opt.hop_length,
			f_max=opt.f_max, f_min=opt.f_min)
	
	def get_frame2mel_idx(self, idx):
		idx = idx - self.num_frames_per_clip//2
		return int(idx * self.bps / self.fps)
	
	def default_img_loader(self, path):
		return cv2.imread(path)[:,:,::-1]

	def default_aud_loader(self, path):
		audio, sr = torchaudio.load(path)
		audio = torch.mean(audio, dim=0)
		if sr != self.sampling_rate:
			audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)(audio)			
			print(f"- [Audio] Resample from {sr} to {self.sampling_rate}")
		mel = self.audio_transform(audio).T
		return torch.log10(torch.clamp(mel, min=1e-5, max=None))
		
	def crop_mel(self, mel, mel_idx, crop_length):
		mel_shape = mel.shape
		if (mel_idx + crop_length) <= mel_shape[0] and mel_idx >= 0:
			mel_cropped = mel[mel_idx:mel_idx + crop_length]
			return mel_cropped
		else:
			if mel_idx < 0:
				pad = -mel_idx
				mel_cropped = F.pad(mel[:mel_idx + crop_length], (0, 0, pad, 0), value=0.)
			else:
				pad = crop_length - (mel_shape[0] - mel_idx)
				mel_cropped = F.pad(mel[mel_idx:], (0, 0, 0, pad), value=0.)
		return mel_cropped
	
	def path2img(self, img_path):
		img = self.default_img_loader(img_path)
		return self.img_transform(image=img)['image']
		
	def get_lower_half_mask(self):
		return NotImplementedError()

	def preprocess(self):
		return NotImplementedError()
	