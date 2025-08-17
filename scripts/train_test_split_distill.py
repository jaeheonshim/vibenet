import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, random_split
from torchvggish import vggish_input
from tqdm import tqdm

from vibenet import labels
from vibenet.dataset import FMAWaveformDataset
from vibenet.models.teacher import PANNsMLP
from vibenet.utils import load

teacher_train_df = pd.read_csv('data/teacher_training/train.csv', index_col=0, header=[0, 1, 2])

val_restrict = set(teacher_train_df.index)

df = load('data/fma_metadata/tracks.csv')
shuffled = df.sample(frac=1, random_state=42)

available_for_val = shuffled.loc[~shuffled.index.isin(val_restrict)]

val_size = int(len(shuffled) * 0.15)
val_df = available_for_val.iloc[:val_size]

remaining = shuffled.drop(val_df.index)
train_df = remaining

train_df.index.to_series().to_csv('data/distillation/train.csv', index=False, header=False)
val_df.index.to_series().to_csv('data/distillation/test.csv', index=False, header=False)