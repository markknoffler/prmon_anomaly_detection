import torch
import torch.nn.functional as F


def sequence_normalised_mse(x, x_hat, eps=1e-8):
    """
    Sequence-normalised MSE.

    Problem with standard MSE:
        A memory-heavy but NORMAL physics job has PSS ~1M kB vs a light job
        at ~150k kB. Standard MSE is dominated by magnitude -- the model learns
        to flag large PSS as anomalous regardless of shape, causing systematic
        false positives on heavy-computation events (a known ATLAS challenge).

    Fix: normalise each feature WITHIN its sequence window (subtract mean,
    divide by std) before computing MSE. Both jobs now have values in ~[-2, 2]
    and the loss measures only how well the temporal SHAPE was reconstructed.

    x, x_hat: (batch, seq_len, n_features)
    """
    mu    = x.mean(dim=1, keepdim=True)
    sigma = x.std(dim=1,  keepdim=True) + eps
    xn    = (x     - mu) / sigma
    xhn   = (x_hat - mu) / sigma   # use INPUT stats for normalising both
    return F.mse_loss(xhn, xn)


def per_sequence_error(x, x_hat, eps=1e-8):
    """Per-sequence scalar error for anomaly scoring. Returns shape (batch,)."""
    mu    = x.mean(dim=1, keepdim=True)
    sigma = x.std(dim=1,  keepdim=True) + eps
    xn    = (x     - mu) / sigma
    xhn   = (x_hat - mu) / sigma
    return ((xhn - xn) ** 2).mean(dim=(1, 2))
