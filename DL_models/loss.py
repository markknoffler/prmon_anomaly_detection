import torch
import torch.nn.functional as F

def per_sequence_error(x, recon):
    return ((x - recon) ** 2).mean(dim=(1, 2))

def mse_loss(x, recon):
    return F.mse_loss(recon, x)
