# src/models/deep_learning/mlp.py
"""
Lightweight Multi-Layer Perceptron (MLP) for time-series buffer forecasting.
Flattens lookback sequence into a 1D vector and projects through a hidden layer with ReLU.
"""
import torch
import torch.nn as nn

class SmallMLP(nn.Module):
    """
    Small MLP baseline mapping [batch_size, seq_len, in_features]
    to [batch_size, 1] for next-step buffer prediction.
    """
    def __init__(self, seq_len: int = 16, in_features: int = 4, hidden_dim: int = 32, out_features: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        flat_dim = seq_len * in_features
        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, in_features]
        b = x.size(0)
        x_flat = x.reshape(b, -1)
        h = self.act(self.fc1(x_flat))
        out = self.fc2(h)
        return out.squeeze(-1)
