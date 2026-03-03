import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """
    Scalar attention weight per timestep over encoder hidden states.
    Distinguishes anomaly shapes (spike) from heavy-normal patterns (plateau)
    because their attention weight distributions differ structurally.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, enc_outputs):
        scores  = self.score(enc_outputs)          # (B, T, 1)
        weights = torch.softmax(scores, dim=1)     # (B, T, 1)
        context = (weights * enc_outputs).sum(1)   # (B, H)
        return context, weights.squeeze(-1)        # context, attn_weights


class TA_LSTM_AE(nn.Module):
    """
    Temporal-Attention LSTM Autoencoder.

    Two upgrades over vanilla LSTM-AE, both addressing the physics false-positive
    problem (heavy-computation jobs triggering spurious anomaly flags):

    (1) Temporal Attention in encoder:
        Standard LSTM-AE keeps only the LAST hidden state. A heavy 1-GB physics
        job has high PSS for its entire duration -- the last state is
        indistinguishable from an anomaly. Attention computes a weighted sum over
        ALL encoder timesteps so the model captures the full temporal shape:
        a uniform plateau (normal heavy job) vs. a sudden spike (anomaly) produce
        structurally different attention weight distributions.

    (2) Sequence-normalised reconstruction loss (see loss.py):
        Decouples shape from magnitude. A 1-GB run and a 150-MB run with the
        same temporal shape produce identical loss -- only shape deviations
        are penalised, not absolute resource levels.
    """
    def __init__(self, n_features, hidden_dim=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_features = n_features

        self.encoder_lstm = nn.LSTM(
            n_features, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim)

        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        B, T, _ = x.shape
        enc_out, (h_n, c_n) = self.encoder_lstm(x)   # (B, T, H)
        context, attn_w     = self.attention(enc_out)  # (B, H), (B, T)
        dec_in  = context.unsqueeze(1).expand(-1, T, -1)  # (B, T, H)
        dec_out, _          = self.decoder_lstm(dec_in, (h_n, c_n))  # (B, T, H)
        recon               = self.output_proj(dec_out)    # (B, T, F)
        return recon, attn_w
