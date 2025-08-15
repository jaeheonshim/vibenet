import os
import bisect
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa
import random
from bisect import bisect_right

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
        self.augment = augment
        self.data, self.labels = [], []

        data_dir = os.path.join(root_dir, 'data')
        labels_dir = os.path.join(root_dir, 'labels')

        files = sorted(os.listdir(data_dir), key=lambda x: int(x.split('.')[0]))
        chunk_lengths = []
        for file in files:
            d = np.load(os.path.join(data_dir, file), mmap_mode='r')
            l = np.load(os.path.join(labels_dir, file), mmap_mode='r')
            self.data.append(d)
            self.labels.append(l)
            chunk_lengths.append(d.shape[0])

        self.starts = np.zeros(len(chunk_lengths), dtype=np.int64)
        if chunk_lengths:
            self.starts[1:] = np.cumsum(chunk_lengths[:-1], dtype=np.int64)

        self.size = int(np.sum(chunk_lengths, dtype=np.int64))

    def __len__(self):
        return self.size

    def _locate(self, idx: int):
        chunk_idx = bisect_right(self.starts, idx) - 1
        local_idx = idx - int(self.starts[chunk_idx])
        return chunk_idx, local_idx

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.size
        if not (0 <= idx < self.size):
            raise IndexError(f'Index {idx} out of range for dataset of size {self.size}')

        chunk_idx, local_idx = self._locate(idx)

        x_np = self.data[chunk_idx][local_idx]
        y_np = self.labels[chunk_idx][local_idx]

        x_np = np.require(x_np, dtype=self.dtype, requirements=['C', 'W'])
        y_np = np.require(y_np, dtype=np.float32, requirements=['C', 'W'])

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)

        return x, y