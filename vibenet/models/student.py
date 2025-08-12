import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvggish import vggish

from torchvision.models import efficientnet_b0
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from vibenet import labels

class EfficientNetRegressor(nn.Module):
    def __init__(self):
        super(EfficientNetRegressor, self).__init__()

        # For getting mel spectrogram from input 16khz waveform
        self.mel = MelSpectrogram(
            sample_rate=16000,
            n_fft=1024, # PANNs
            hop_length=320, # PANNs
            win_length=640,
            f_max=8000, # Human hearing limit
            power=2,
            n_mels=128,
            center=True,
            pad_mode='reflect',
            norm='slaney',
            mel_scale='slaney'
        )

        # Transform spectrogram to logarithmic scale
        self.to_db = AmplitudeToDB(stype="power", top_db=80)

        # Backbone model for embeddings
        self.backbone = efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity() # Remove the classifer

        # Trunk to reduce dimension for output heads
        self.trunk = nn.Sequential(
            nn.LayerNorm(1280),
            nn.Linear(1280, 256),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.heads = nn.ModuleDict({n: nn.Linear(256, 1) for n in labels})

    def forward(self, x: torch.Tensor):
        x = self.mel(x)
        x = self.to_db(x)

        x = x.unsqueeze(1)
        x = x.repeat(1, 3, 1, 1) # (batch_size, 3, n_mels, time)

        x = self.backbone(x)
        x = self.trunk(x)

        out = {k: self.heads[k](x).squeeze(-1) for k in self.heads}
        
        return out