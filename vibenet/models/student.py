import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvggish import vggish
from torchvision.models import efficientnet_b0

from vibenet import labels


class EfficientNetRegressor(nn.Module):
    def __init__(self):
        super(EfficientNetRegressor, self).__init__()

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
        x = x.unsqueeze(1)
        x = x.repeat(1, 3, 1, 1) # (batch_size, 3, n_mels, time)

        x = self.backbone(x)
        x = self.trunk(x)

        outs = [self.heads[n](x) for n in labels]
        out = torch.cat(outs, dim=1)
        
        return out