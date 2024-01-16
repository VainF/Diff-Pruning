import torch
import random, os
import argparse
from PIL import Image
import torchvision
import numpy as np
import pytorch_msssim
from utils import UnlabeledImageFolder
from tqdm import tqdm 
import torch_pruning as tp
parser = argparse.ArgumentParser()
parser.add_argument('--restore_from', type=str, required=True)
args = parser.parse_args()

model = torch.load(args.restore_from, map_location='cpu')[0]
example_inputs = {'x': torch.randn(1, 3, 32, 32), 't': torch.ones(1)}
macs, params = tp.utils.count_ops_and_params(model, example_inputs)
print("model: {}, macs: {} G, params: {} M".format(args.restore_from, macs/1e9, params/1e6))

