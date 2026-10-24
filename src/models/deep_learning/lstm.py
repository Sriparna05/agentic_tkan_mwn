# models/deep_learning/lstm.py
"""
Standard LSTM architecture for multi-step time-series forecasting.
"""
import torch
import torch.nn as nn

class StandardLSTM(nn.Module):
    """
    Standard multi-layer LSTM model projecting the final recurrent state
    to the target prediction horizon.
    """
    def __init__(self, input_dim: int, horizon: int = 96, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [Batch, Lookback (96), Features]
        out, _ = self.lstm(x)  # out shape: [Batch, Lookback, Hidden]
        # Gather final hidden state to predict the sequence
        out = out[:, -1, :]  # out shape: [Batch, Hidden]
        out = self.fc(out)   # out shape: [Batch, Horizon]
        return out.unsqueeze(-1)  # out shape: [Batch, Horizon, 1]