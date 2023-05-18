import torch
from glob import glob
from PIL import Image
import os
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, CIFAR100

class UnlabeledImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, exts=["*.jpg", "*.png", "*.jpeg", "*.webp"]):
        self.root = root
        self.files = []
        self.transform = transform
        for ext in exts:
            self.files.extend(glob(os.path.join(root, '**/*.{}'.format(ext)), recursive=True))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

def set_dropout(model, p):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = p

def get_dataset(name_or_path, transform=None):
    if name_or_path.lower()=='cifar10':
        if transform is None:
            transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
            ])
        dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif name_or_path.lower()=='cifar100':
        if transform is None:
            transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
            ])
        dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    elif os.path.isdir(name_or_path):
        if transform is None:
            transform = T.Compose([
                T.Resize(256),
                T.RandomCrop(256),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
            ])
        dataset = UnlabeledImageFolder(name_or_path, transform=transform)
    return dataset

