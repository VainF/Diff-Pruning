import os
import torchvision
from torchvision.datasets import CIFAR10
from tqdm import tqdm

# Define the path to the folder where the images will be saved
save_path = 'data/cifar10/images'

# Create the folder if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load the CIFAR10 dataset
dataset = CIFAR10(root='data/cifar10', train=True, download=True)

# Loop through the dataset and save each image to the folder
for i in tqdm(range(len(dataset))):
    image, label = dataset[i]
    image_name = f'{i}.png'
    image_path = os.path.join(save_path, image_name)
    image.save(image_path)