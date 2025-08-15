from vibenet.pann import models as pann_models
from vibenet.utils import do_mixup
from vibenet import labels

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvggish import vggish

class PANNsMLP(nn.Module):
    def __init__(self):
        super(PANNsMLP, self).__init__()
        
        checkpoint = torch.load('checkpoints/Cnn14_mAP=0.431.pth')
        self.pann = pann_models.Cnn14(32000, 1024, 320, 64, 50, 14000, 527)
        self.pann.load_state_dict(checkpoint['model'], strict=False)
        
        self.trunk = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.heads = nn.ModuleDict({n: nn.Linear(256, 1) for n in labels})
        
        # Freeze all PANN parameters
        for p in self.pann.parameters():
            p.requires_grad = False
            
        # Unfreeze last layer of PANN
        for m in self.pann.conv_block6.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False
                    m.eval()
            else:
                for p in m.parameters():
                    p.requires_grad = True

        
    def forward(self, input):
        output = self.pann(input)
        embeddings = output['embedding']
            
        h = self.trunk(embeddings)
        out = {k: self.heads[k](h).squeeze(-1) for k in self.heads}
        return out

class VGGishMLP(nn.Module):
    def __init__(self):
        super(VGGishMLP, self).__init__()

        self.vgg = vggish(postprocess=False)
        self.vgg.eval()

        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 7)

    def forward(self, input):
        B, T, H, W = input.shape

        data = input.reshape(B*T, 1, H, W)
        with torch.inference_mode():
            embeddings = self.vgg(data)
        embeddings = embeddings.reshape(B, T, -1)


        mean = embeddings.mean(dim=1)
        std = embeddings.std(dim=1)

        x = torch.cat([mean, std], dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.sigmoid(self.fc4(x))

        return x


class VGGishLSTM(nn.Module):
    def __init__(self):
        super(VGGishLSTM, self).__init__()

        self.vgg = vggish(postprocess=False)
        self.vgg.eval()

        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 7)

    def forward(self, input, mixup_lambda=None):
        B, T, H, W = input.shape

        data = input.reshape(B*T, 1, H, W)
        with torch.inference_mode():
            embeddings = self.vgg(data)
        embeddings = embeddings.reshape(B, T, -1)

        embeddings = embeddings.clone().detach()
        embeddings = embeddings.view(B, T, 128)

        if self.training and mixup_lambda is not None:
            embeddings += 0.01 * torch.randn_like(embeddings) # Gaussian noise
            embeddings = do_mixup(embeddings, mixup_lambda)

        lstm_out, _ = self.lstm(embeddings)

        x = lstm_out[:,-1]
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.sigmoid(self.fc2(x))

        return x