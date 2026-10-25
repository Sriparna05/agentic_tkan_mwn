# models/kan/tkan.py
"""
PyTorch-native mathematically identical implementation of the TKAN (Temporal KAN).
Directly transcribed from the original Keras 3 backend-agnostic source to interface 
with the PyTorch-based Phase 2.6 benchmark pipeline.
"""
import torch
import torch.nn as nn
import math

# Import the core spline mathematics from our Vanilla KAN implementation
try:
    from src.models.kan.vanilla_kan import KANLayer
except ImportError:
    from models.kan.vanilla_kan import KANLayer



class TKANCell(nn.Module):
    """
    PyTorch implementation of TKANCell. Retains exact gate mechanics, sub-KAN aggregations, 
    and parameter dimension allocations from the original Keras 3 specification.
    """
    def __init__(
        self, 
        input_dim: int, 
        units: int, 
        num_sub_kans: int = 1, 
        sub_kan_input_dim: int = None, 
        sub_kan_output_dim: int = None,
        grid_size: int = 5,
        k: int = 3
    ):
        super().__init__()
        self.units = units
        self.num_sub_kans = num_sub_kans
        self.sub_kan_input_dim = sub_kan_input_dim if sub_kan_input_dim is not None else input_dim
        self.sub_kan_output_dim = sub_kan_output_dim if sub_kan_output_dim is not None else input_dim

        # 1. Main LSTM-style Gates (kernel, recurrent_kernel, bias)
        self.kernel = nn.Parameter(torch.empty(input_dim, units * 3))
        self.recurrent_kernel = nn.Parameter(torch.empty(units, units * 3))
        self.bias = nn.Parameter(torch.empty(units * 3))

        # 2. Sub-KAN Network Kernels
        self.tkan_sub_layers = nn.ModuleList([
            KANLayer(self.sub_kan_input_dim, self.sub_kan_output_dim, grid_size, k) 
            for _ in range(num_sub_kans)
        ])

        self.sub_tkan_kernel = nn.Parameter(torch.empty(num_sub_kans, self.sub_kan_output_dim * 2))
        self.sub_tkan_recurrent_kernel_inputs = nn.Parameter(torch.empty(num_sub_kans, input_dim, self.sub_kan_input_dim))
        self.sub_tkan_recurrent_kernel_states = nn.Parameter(torch.empty(num_sub_kans, self.sub_kan_output_dim, self.sub_kan_input_dim))

        # 3. Aggregation mechanism
        self.aggregated_weight = nn.Parameter(torch.empty(num_sub_kans * self.sub_kan_output_dim, units))
        self.aggregated_bias = nn.Parameter(torch.empty(units))

        self.reset_parameters()

    def reset_parameters(self):
        """Initializations mimicking standard Keras behavior (Glorot/Orthogonal/Forget Bias)."""
        nn.init.xavier_uniform_(self.kernel)
        nn.init.orthogonal_(self.recurrent_kernel)
        nn.init.zeros_(self.bias)
        # Keras unit_forget_bias = True (Middle chunk corresponds to the forget gate)
        nn.init.ones_(self.bias[self.units : 2 * self.units])
        
        nn.init.orthogonal_(self.sub_tkan_kernel)
        nn.init.orthogonal_(self.sub_tkan_recurrent_kernel_inputs)
        nn.init.orthogonal_(self.sub_tkan_recurrent_kernel_states)
        nn.init.xavier_uniform_(self.aggregated_weight)
        nn.init.zeros_(self.aggregated_bias)

    def init_states(self, batch_size: int, device: torch.device):
        """Returns initialized hidden, cell, and sub-KAN states."""
        h = torch.zeros(batch_size, self.units, device=device)
        c = torch.zeros(batch_size, self.units, device=device)
        sub_states = [torch.zeros(batch_size, self.sub_kan_output_dim, device=device) for _ in range(self.num_sub_kans)]
        return [h, c] + sub_states

    def forward(self, inputs: torch.Tensor, states: list) -> tuple:
        h_tm1, c_tm1 = states[0], states[1]
        sub_states = states[2:]

        # Standard LSTM Gate Processing
        gates = torch.matmul(inputs, self.kernel) + torch.matmul(h_tm1, self.recurrent_kernel) + self.bias
        # The Keras implementation splits the sigmoid activation of the gates into 3 parts (i, f, c)
        i, f, c_cand = torch.split(torch.sigmoid(gates), self.units, dim=-1)

        c = f * c_tm1 + i * c_cand

        # Process each KAN sub-layer
        sub_outputs = []
        new_sub_states = []

        for idx, (sub_layer, sub_state) in enumerate(zip(self.tkan_sub_layers, sub_states)):
            sub_kernel_x = self.sub_tkan_recurrent_kernel_inputs[idx]
            sub_kernel_h = self.sub_tkan_recurrent_kernel_states[idx]

            agg_input = torch.matmul(inputs, sub_kernel_x) + torch.matmul(sub_state, sub_kernel_h)
            sub_output = sub_layer(agg_input)

            sub_recurrent_kernel_h, sub_recurrent_kernel_x = torch.split(self.sub_tkan_kernel[idx], self.sub_kan_output_dim, dim=-1)
            new_sub_state = sub_recurrent_kernel_h * sub_output + sub_state * sub_recurrent_kernel_x

            sub_outputs.append(sub_output)
            new_sub_states.append(new_sub_state)

        # Aggregate Sub-layer outputs
        aggregated_sub_output = torch.cat(sub_outputs, dim=-1)
        aggregated_input = torch.matmul(aggregated_sub_output, self.aggregated_weight) + self.aggregated_bias

        xo = torch.sigmoid(aggregated_input)

        # Compute cell's final hidden state
        h = xo * torch.tanh(c)

        return h, [h, c] + new_sub_states


class TKAN(nn.Module):
    """
    Standard PyTorch RNN Wrapper for the TKANCell for time-series forecasting.
    Iterates sequentially through the lookback window to map to the prediction horizon.
    """
    def __init__(
        self, 
        input_dim: int, 
        horizon: int = 96, 
        units: int = 64, 
        num_sub_kans: int = 2,
        grid_size: int = 5,
        k: int = 3
    ):
        super().__init__()
        self.cell = TKANCell(input_dim, units, num_sub_kans=num_sub_kans, grid_size=grid_size, k=k)
        self.fc = nn.Linear(units, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [Batch, Sequence_Len, Features]
        batch_size, seq_len, _ = x.shape
        states = self.cell.init_states(batch_size, x.device)

        # Unroll RNN sequence over the lookback window
        for t in range(seq_len):
            h, states = self.cell(x[:, t, :], states)

        # Project final recurrent step state to target output horizon
        out = self.fc(h)
        return out.unsqueeze(-1)  # Output shape: [Batch, Horizon, 1]