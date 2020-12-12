import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((120 , 120 ), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files_hr = sorted(glob.glob(root + "hr/*.*"))
        self.files_lr = sorted(glob.glob(root + "lr/*.*"))



    def __getitem__(self, index):
        img_pre_hr = Image.open(self.files_hr[index % len(self.files_hr)])
        img_pre_lr = Image.open(self.files_lr[index % len(self.files_lr)])

        img_lr = self.lr_transform(img_pre_lr)
        img_hr = self.hr_transform(img_pre_hr)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files_hr)
