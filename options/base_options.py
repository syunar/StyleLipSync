import os, pickle, json, argparse

def save_options(opt, save_path):
    with open(save_path, 'wt') as f:
        json.dump(vars(opt), f, indent=4)

def load_options(opt, load_path):
    with open(load_path, 'rt') as f:
        _update = json.loads(f)
    opt.update(_update)
    return opt

class BaseOptions():
    def parse(self):
        parser = argparse.ArgumentParser()
        self.parser = self.initialize(parser)
        self.opt = self.parser.parse_args()
        return self.opt

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--name', type=str, help='name of the experiment. It decides where to store samples and models')
        # input setting
        parser.add_argument('--input_size', type=int, default=256, help='input image size')
        parser.add_argument('--input_nc', type=int, default=3, help='input image channel')
        parser.add_argument('--fps', type=int, default=25)
        parser.add_argument('--num_frames_per_clip', type=int, default=5)

        # audio preprocessing
        parser.add_argument('--sampling_rate', type=int, default=16000)
        parser.add_argument("--n_mels", type=int, default=80)
        parser.add_argument("--n_fft", type=int, default=512)
        parser.add_argument("--win_length", type=int, default=400)
        parser.add_argument("--hop_length", type=int, default=160)
        parser.add_argument("--f_max", type=float, default=7600.)
        parser.add_argument("--f_min", type=float, default=55.)
        return parser

    def print_options(self):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
