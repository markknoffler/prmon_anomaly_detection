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
        scores  = self.score(enc_outputs)          
        weights = torch.softmax(scores, dim=1)     
        context = (weights * enc_outputs).sum(1)   
        return context, weights.squeeze(-1)     


class TA_LSTM_AE(nn.Module):
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
