from __future__ import print_function, division
import os
import torch
import pandas as pd # 用于更容易地进行csv解析
from skimage import io, transform # 用于图像的IO和变换
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision_learn import transforms, utils, datasets


data_transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    # transforms.RandomResizedCrop(224)
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset, batch_size=4, shuffle=True, num_workers=4)


