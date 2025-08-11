import os
import bisect
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa
import random

CHUNK_SIZE = 1000

class FMAVGGishDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.data = []
        self.labels = []

        self.size = 0

        for file in sorted(os.listdir(os.path.join(root_dir, 'data')), key=lambda x: int(x.split('.')[0])):
            d = np.load(os.path.join(root_dir, 'data', file), mmap_mode='r')
            l = np.load(os.path.join(root_dir, 'labels', file), mmap_mode='r')
            self.data.append(d)
            self.labels.append(l)

            self.size += d.shape[0]

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        chunk_idx = idx // CHUNK_SIZE
        local_idx = idx % CHUNK_SIZE

        return self.data[chunk_idx][local_idx], self.labels[chunk_idx][local_idx]

class FMAWaveformDataset(Dataset):
    def __init__(self, root_dir, dtype=np.float32, augment=False):
        self.dtype = dtype
        self.data, self.labels, self.size = [], [], 0

        for file in sorted(os.listdir(os.path.join(root_dir, 'data')), key=lambda x: int(x.split('.')[0])):
            d = np.load(os.path.join(root_dir, 'data', file), mmap_mode='r')
            l = np.load(os.path.join(root_dir, 'labels', file), mmap_mode='r')
            self.data.append(d); self.labels.append(l)
            self.size += d.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        chunk_idx = idx // CHUNK_SIZE
        local_idx = idx % CHUNK_SIZE

        x_np = self.data[chunk_idx][local_idx]
        y_np = self.labels[chunk_idx][local_idx]

        x_np = np.require(x_np, dtype=self.dtype, requirements=['C', 'W'])
        y_np = np.require(y_np, dtype=np.float32, requirements=['C', 'W'])

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        return x, y