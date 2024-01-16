import torch
import random, os
import argparse
from PIL import Image
import torchvision
import numpy as np
import pytorch_msssim
from utils import UnlabeledImageFolder
from tqdm import tqdm 
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, nargs='+')
args = parser.parse_args()

# generate radom index
nrow = 16
img_index = random.sample(list(range(50000)), nrow*nrow)
path1 = args.path[0]
path2 = args.path[1]
print(path1, path2)
img_dst1 = UnlabeledImageFolder(path1, transform=torchvision.transforms.ToTensor(), exts=["png"])
img_dst2 = UnlabeledImageFolder(path2, transform=torchvision.transforms.ToTensor(), exts=["png"])
print(len(img_dst1), len(img_dst2))

loader1 = torch.utils.data.DataLoader(
    img_dst1,
    batch_size=100,
    shuffle=False,
    num_workers=4,
    drop_last=False,
)
loader2 = torch.utils.data.DataLoader(
    img_dst2,
    batch_size=100,
    shuffle=False,
    num_workers=4,
    drop_last=False,
)

with torch.no_grad():
    ssim_list = []
    mse_list = []
    for i, (img1, img2) in tqdm(enumerate(zip(loader1, loader2))):
        ssim = pytorch_msssim.ssim(img1.cuda(), img2.cuda(), data_range=1.0, size_average=False)
        ssim_list.append(ssim.cpu())
        mse = torch.nn.functional.mse_loss(img1.cuda(), img2.cuda(), reduction='none').mean(dim=(1,2,3))
        mse_list.append(mse.cpu())

    ssim = torch.cat(ssim_list, dim=0)
    mse = torch.cat(mse_list, dim=0)
    ssim_avg = ssim.mean()
    mse_avg = mse.mean()
    print("path1: {}, path2: {}, ssim: {}, mse: {}".format(path1, path2, ssim_avg, mse_avg))

    