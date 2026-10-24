# models/kan/vanilla_kan.py
"""
Unmodified mathematical core of Vanilla KAN (Kolmogorov-Arnold Network).
Stripped of symbolic regression, pruning, and visualization overhead to 
function purely as a standard deep learning baseline for time-series benchmarking.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    """
    Core Kolmogorov-Arnold Layer computing learnable 1D B-splines + SiLU residual.
    """
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5, k: int = 3, 
                 noise_scale: float = 0.1, scale_base: float = 1.0, scale_sp: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.k = k
        self.scale_base = scale_base
        self.scale_sp = scale_sp
        self.base_fun = nn.SiLU()

        # Mathematically strict grid generation to prevent floating-point drift
        step = 2.0 / grid_size
        num_grid_points = grid_size + 2 * k + 1
        grid = torch.linspace(-1 - k * step, 1 + k * step, num_grid_points).expand(in_dim, -1).clone()
        self.register_buffer('grid', grid)

        # Spline Coefficients
        self.coef = nn.Parameter(torch.Tensor(out_dim, in_dim, grid_size + k))
        nn.init.normal_(self.coef, mean=0.0, std=noise_scale / (math.sqrt(grid_size)))

        # Base function linear weights
        self.w_base = nn.Parameter(torch.Tensor(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.w_base, a=math.sqrt(5))

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the B-spline basis functions recursively.
        """
        x = x.unsqueeze(-1)  # [Batch, in_dim, 1]
        grid = self.grid     # [in_dim, grid_size + 2k + 1]
        
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()
        for p in range(1, self.k + 1):
            left_num = x - grid[:, :-(p + 1)]
            left_den = grid[:, p:-1] - grid[:, :-(p + 1)]
            left = torch.where(left_den == 0, torch.zeros_like(left_num), left_num / left_den)
            
            right_num = grid[:, p + 1:] - x
            right_den = grid[:, p + 1:] - grid[:, 1:-p]
            right = torch.where(right_den == 0, torch.zeros_like(right_num), right_num / right_den)
            
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases  # [Batch, in_dim, grid_size + k]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual Base: SiLU(x) * w_base
        base_out = F.linear(self.base_fun(x), self.w_base)  # [Batch, out_dim]
        
        # Splines: B-spline bases * coefficients
        splines = self.b_splines(x)  # [Batch, in_dim, grid_size + k]
        spline_out = torch.einsum('bik,oik->bo', splines, self.coef)  # [Batch, out_dim]
        
        return self.scale_base * base_out + self.scale_sp * spline_out


class VanillaKAN(nn.Module):
    """
    Standard MLP-style Kolmogorov-Arnold Network for Time-Series Forecasting.
    Flattens the temporal lookback into spatial dimensions to map 
    (Lookback * Features) -> (Horizon).
    """
    def __init__(self, seq_len: int, features: int, horizon: int = 96, 
                 hidden_dim: int = 64, num_layers: int = 2, grid_size: int = 5, k: int = 3):
        super().__init__()
        self.seq_len = seq_len
        self.features = features
        self.horizon = horizon
        
        input_dim = seq_len * features
        self.layers = nn.ModuleList()
        
        # Input Layer
        self.layers.append(KANLayer(input_dim, hidden_dim, grid_size, k))
        # Hidden Layers
        for _ in range(num_layers - 1):
            self.layers.append(KANLayer(hidden_dim, hidden_dim, grid_size, k))
        # Output Layer
        self.layers.append(KANLayer(hidden_dim, horizon, grid_size, k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [Batch, Lookback, Features]
        B, L, F = x.shape
        # Flatten spatial-temporal dimensions for standard KAN mapping
        x = x.view(B, L * F)
        
        for layer in self.layers:
            x = layer(x)
            
        return x.unsqueeze(-1)  # Output shape: [Batch, Horizon, 1]