import os, glob, math, shutil, subprocess, pickle, copy, yaml, random, json, cv2

import torch, torchaudio, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm

import albumentations as A
import albumentations.pytorch.transforms as A_pytorch


from utils import common
from options.train_options import TrainOptions
from dataloader.base_dataloader import BaseDataLoader


class InferenceOptions(TrainOptions):
	def __init__(self):
		super().__init__()
	def initialize(self, parser):
		super().initialize(parser)
		parser.add_argument("--audio_path", type=str,
			default=None, required=True)
		parser.add_argument("--person", type=str,
			default="AlexandriaOcasioCortez_0")
		parser.add_argument("--ckpt", type=str,
			default="./ckpts/AlexandriaOcasioCortez_0_3k.pth")
		parser.add_argument("--w_avg_path", type=str,
			default="./ckpts/w_avg.pth",)
		parser.add_argument("--res_dir", type=str,
			default="./results")
		return parser

class InferenceDataLoader(BaseDataLoader):
	def __init__(self, opt):
		super().__init__(opt)

	def load_frame_and_mask(self, frame_path, mask_path):
		frames = sorted(glob.glob(os.path.join(frame_path, "*.jpg")))
		masks  = sorted(glob.glob(os.path.join(mask_path, "*.jpg")))
		return frames, masks

	def get_lower_half_mask(self, img_path, mask_path):
		mask = cv2.imread(mask_path)
		mask = cv2.dilate(mask, kernel = np.ones((3, 3), np.uint8), iterations=9)
		mask_pti = mask / 255.
		mask = 255 - mask
		mask = cv2.GaussianBlur(mask, (21, 21), 0) 
		img = self.default_img_loader(img_path)
		mask = mask / 255.
		img_mask = img * mask
		img_mask = self.img_transform(image=img_mask)['image']
		return img_mask, mask

	def preprocess(self, audio_path, frame_path, mask_path) -> dict:
		mel = self.default_aud_loader(audio_path)
		max_idx = int(mel.shape[0] * self.fps / self.bps)
		frame_pathes, mask_pathes = self.load_frame_and_mask(frame_path, mask_path)
		frame_pathes = 10*(frame_pathes + list(reversed(frame_pathes)))
		mask_pathes  = 10*(mask_pathes + list(reversed(mask_pathes)))
		mel_batch, x_batch, ref_batch, mask_batch, target_batch = [], [], [], [], []
		for idx in range(max_idx):
			mel_idx = self.get_frame2mel_idx(idx)
			mel_   = self.crop_mel(mel, mel_idx, self.num_mel_bins)
			img_mask, mask_ = self.get_lower_half_mask(frame_pathes[idx], mask_pathes[idx])
			target = self.default_img_loader(frame_pathes[idx])
			target_batch.append(target)
			# mask_ = mask_ / 255.
			x_batch.append(img_mask)
			mel_batch.append(mel_)
			mask_batch.append(mask_)

			if idx == 0: # use the first frame as the reference image
				ref_ = self.img_transform(image=target)['image']
				ref_batch.append(ref_)
		x_batch   = torch.stack(x_batch, dim=0).unsqueeze(0)
		mel_batch = torch.stack(mel_batch, dim=0).unsqueeze(0)
		ref_batch = torch.stack(ref_batch, dim=0).unsqueeze(0)
		return {'x': x_batch, 'mel': mel_batch, 'ref': ref_batch, 'mask': mask_batch, 'target': target_batch}


class Demo:
	def __init__(self, opt=None, rank=None):
		torch.cuda.empty_cache()
		self.opt = opt
		self.rank = rank

		self.init_network()
		self.load_model_snapshot(opt.ckpt, self.rank)

	def init_network(self):
		from models.models_stylelipsync.stylelipsync import StyleLipSync
		self.G = StyleLipSync(self.opt)

	def load_model_snapshot(self, snapshot_path, distributed=False):
		checkpoint_dict = torch.load(snapshot_path, map_location='cpu')
		if 'G' in checkpoint_dict:
			state_dict = {}
			for k, v in checkpoint_dict['G'].items():
				state_dict[k.replace('module.', '')] = v
			self.G.load_state_dict(state_dict)
			print(f"- ckpt loaded from {snapshot_path}")
		else:
			print(f"- failed to load ckpt from {snapshot_path}")
		self.G.to(self.rank)	
		self.G.eval()
	
	def frames_to_video(self, x_hat, mask, target, video_path:str, audio_path:str) -> str:
		tmp_video_path = video_path[:-4] + "_tmp.mp4"		
		frames = common.batch2im(x_hat)
		vw = cv2.VideoWriter(tmp_video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), self.opt.fps, (self.opt.input_size, self.opt.input_size))
		for idx in tqdm(range(len(frames)), total=len(frames)):
			frame = frames[idx][...,::-1]
			target_ = target[idx][...,::-1]
			mask_   = mask[idx]
			frame = frame * (1-mask_) + target_ * mask_
			vw.write(frame.astype(np.uint8))
		vw.release()
		video_path = self.merge_audio_to_video(tmp_video_path, video_path, audio_path)
		return video_path

	def merge_audio_to_video(self, tmp_video_path:str, video_path:str, audio_path=None) -> str:
		if audio_path is not None:
			with open(os.devnull, 'wb') as f:
				command = ("ffmpeg -y -loglevel panic -i %s -i %s -q:v 2 -strict -2 %s" % \
								(tmp_video_path, audio_path, video_path)) # Combine audio and video file
				subprocess.call(command, shell=True, stdout=f, stderr=f)
			if os.path.exists(video_path): os.remove(tmp_video_path)
		else:
			with open(os.devnull, 'wb') as f:
				command = ("mv %s %s" % (tmp_video_path, video_path)) # rename the video file
				subprocess.call(command, shell=True, stdout=f, stderr=f)
		return video_path

	@torch.no_grad()
	def run(self, data, video_path, audio_path):
		x_hat = self.G.inference(data)['x_hat']
		video_path = self.frames_to_video(x_hat, data['mask'], data['target'], video_path, audio_path)
		return video_path

if __name__ == '__main__':
	parser = InferenceOptions()
	opt = parser.parse()
	os.makedirs(opt.res_dir, exist_ok=True)

	audio_path = opt.audio_path
	person = opt.person
	frame_path = os.path.join("./data", f"{person}", "frame")
	mask_path  = os.path.join("./data", f"{person}", "mask")

	dataloader = InferenceDataLoader(opt)
	data = dataloader.preprocess(audio_path, frame_path, mask_path)
	
	demo = Demo(opt, rank=opt.rank)
	video_path = os.path.join(opt.res_dir, f"{opt.person}#{os.path.basename(opt.audio_path)[:-4]}.mp4")
	demo.run(data, video_path, opt.audio_path)

