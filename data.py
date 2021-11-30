import os
import torch
import numpy as np
from torchvision import transforms,utils
from torch.utils.data import Dataset,DataLoader
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, txt, transforms, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transforms
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]     
        img = self.loader('/home/ubuntu/erick/FGVC/CUB_200_2011/images/' + fn)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)