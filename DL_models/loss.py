import torch
import torch.nn.functional as F


def sequence_normalised_mse(x, x_hat, eps=1e-8):
    return F.mse_loss(x_hat, x)


def per_sequence_error(x, x_hat, eps=1e-8):
    """Per-sequence scalar error for anomaly scoring. Returns shape (batch,)."""
    return ((x_hat - x) ** 2).mean(dim=(1, 2))

