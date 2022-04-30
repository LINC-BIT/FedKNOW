# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/Update.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
import time
import copy
import FedProx

from models.language_utils import get_word_emb_arr, repackage_hidden, process_x, process_y 
from torchvision import transforms
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, name=None):
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        self.dataset = dataset
        self.idxs = list(idxs)
        self.name = name
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.name is None:
            image, label = self.dataset[self.idxs[item]]
        elif 'femnist' in self.name:
            image = torch.reshape(torch.tensor(self.dataset['x'][item]),(1,28,28))
            label = torch.tensor(self.dataset['y'][item])
        elif 'sent140' in self.name:
            image = self.dataset['x'][item]
            label = self.dataset['y'][item]
        elif 'cifar100' in self.name:
            image, label = self.dataset[self.idxs[item]][0], self.dataset[self.idxs[item]][1]
            image = Image.fromarray(image)
            if self.transform is not None:
                image = self.transform(image)
        elif 'miniimagenet' in self.name:
            image,label = self.dataset[self.idxs[item]][0],self.dataset[self.idxs[item]][1]
        return image, label



