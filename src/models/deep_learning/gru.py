# models/deep_learning/gru.py
"""
Standard Gated Recurrent Unit (GRU) architecture for multi-step time-series forecasting.
"""
import torch
import torch.nn as nn

class StandardGRU(nn.Module):
    """
    Standard multi-layer GRU model. Maps recurrent representations to
    the designated forecasting horizon.
    """
    def __init__(self, input_dim: int, horizon: int = 96, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [Batch, Lookback, Features]
        out, _ = self.gru(x)  # out shape: [Batch, Lookback, Hidden]
        # Gather final sequence step
        out = out[:, -1, :]  # out shape: [Batch, Hidden]
        out = self.fc(out)   # out shape: [Batch, Horizon]
        return out.unsqueeze(-1)  # out shape: [Batch, Horizon, 1]


class SingleLayerGRU(nn.Module):
    """
    Lightweight single-layer GRU baseline for next-sample buffer forecasting.
    Maps [batch_size, seq_len, in_features] to [batch_size] scalar predictions.
    """
    def __init__(self, in_features: int = 4, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)