import torch
import random, os
import argparse
from PIL import Image
import torchvision
import numpy as np
import pytorch_msssim
from utils import UnlabeledImageFolder
from tqdm import tqdm 
img_ids = [159, 149, 144, 127, 86, 41]
image_folder1 = 'run/sample_v2/bedroom_250k/image_samples/images/0'
image_folder2 = 'run/sample_v2/bedroom_official/image_samples/images/0'
base_img_id = 0
ssim_list = []
for iid in img_ids:
    img1 = Image.open(os.path.join(image_folder1, f'{iid}.png'))
    img2 = Image.open(os.path.join(image_folder2, f'{iid}.png'))
    img1_tensor = torchvision.transforms.ToTensor()(img1).unsqueeze(0)
    img2_tensor = torchvision.transforms.ToTensor()(img2).unsqueeze(0)
    ssim = pytorch_msssim.ssim(img1_tensor, img2_tensor, data_range=1.0, size_average=True)
    ssim_list.append(ssim.item())
print(ssim_list)
    