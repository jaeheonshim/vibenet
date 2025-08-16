import torch
from torch import nn
import torch.nn.functional as F
from vibenet import labels, LIKELIHOODS, CONTINUOUS

huber = nn.SmoothL1Loss(beta=0.2)
mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()

def pearsonr(x, y, eps=1e-8):
    x = x - x.mean()
    y = y - y.mean()
    return (x*y).mean() / (x.std(unbiased=False)*y.std(unbiased=False) + eps)

def ccc(x, y, eps=1e-8):
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(unbiased=False), y.var(unbiased=False)
    cov = ((x-mx)*(y-my)).mean()
    return (2*cov) / (vx + vy + (mx-my).pow(2) + eps)

def compute_losses(pred, label):
    return {
        'acousticness_bce': bce(pred['acousticness'], label[:,0]),
        'liveness_bce': bce(pred['liveness'], label[:,4]),
        'instrumentalness_bce': bce(pred['instrumentalness'], label[:,3]),
        'speechiness_huber': huber(pred['speechiness'], label[:,5]),
        'danceability_mse': mse(pred['danceability'], label[:,1]),
        'energy_ccc_loss': 1 - ccc(pred['energy'], label[:,2]),
        'valence_ccc_loss': 1 - ccc(pred['valence'], label[:,6]),
    }

def compute_metrics(pred, label):
    out = {}
    for i, name in enumerate(labels):
        y = label[:, i]
        yhat = pred[name].squeeze(-1) if pred[name].ndim > 1 else pred[name]
        m = {}
        
        if name in LIKELIHOODS:
            m['logloss'] = bce(yhat, y)
        else:
            mse_v = F.mse_loss(yhat, y)
            mae_v = torch.mean(torch.abs(yhat - y))
            m['mse'] = mse_v
            m['rmse'] = torch.sqrt(mse_v)
            m['mae'] = mae_v
            m['pearson'] = pearsonr(yhat, y)
            m['ccc'] = ccc(yhat, y)
            var_y = y.var(unbiased=False) + 1e-8
            m['r2'] = 1.0 - (mse_v / var_y)

        out[name] = {k: (v.item() if torch.is_tensor(v) else float(v)) for k, v in m.items()}
        
    return out