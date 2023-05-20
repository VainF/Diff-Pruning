import pytorch_msssim 
import os
import torch
from PIL import Image
import torchvision

base_folder_name = 'run/prune_ssim_2/0'
folder_name = [os.path.join('run/prune_ssim_2', '{}'.format(k)) for k in range(50, 1000+1, 50)]
n_samples = 32
# test ssim for each folder
folder_ssim = []
for f in folder_name:
    ssim_list = []
    for img_id in range(n_samples):
        img1 = Image.open(os.path.join(base_folder_name, f'{img_id}.png'))
        img2 = Image.open(os.path.join(f, f'{img_id}.png'))
        img1_tensor = torchvision.transforms.ToTensor()(img1)
        img2_tensor = torchvision.transforms.ToTensor()(img2)
        img1_tensor = img1_tensor.unsqueeze(0)
        img2_tensor = img2_tensor.unsqueeze(0)
        ssim = pytorch_msssim.ssim(img1_tensor, img2_tensor, data_range=1.0, size_average=True)
        ssim_list.append(ssim)
    ssim = sum(ssim_list) / len(ssim_list)
    folder_ssim.append(ssim.item())
print(folder_ssim)