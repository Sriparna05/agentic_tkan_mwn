import torch
import torch.nn as nn
from models.kan.vanilla_kan import KANLayer


class AgenticTKANCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, grid_size: int = 5, k: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = 1
        concat_dim = input_dim + self.action_dim + hidden_dim

        # Step 4 & 5: Internal Memory & Context (KAN-based LSTM Gates)
        self.kan_i = KANLayer(concat_dim, hidden_dim, grid_size=grid_size, k=k)
        self.kan_f = KANLayer(concat_dim, hidden_dim, grid_size=grid_size, k=k)
        self.kan_c = KANLayer(concat_dim, hidden_dim, grid_size=grid_size, k=k)
        self.kan_o = KANLayer(concat_dim, hidden_dim, grid_size=grid_size, k=k)

        # Step 6: Joint Agentic Heads
        # Branch A: Internal Decision / Policy Module
        self.kan_policy = KANLayer(hidden_dim, self.action_dim, grid_size=grid_size, k=k)

        # Branch B: Internal Self-Eval / Critic
        self.kan_critic = KANLayer(hidden_dim, self.action_dim, grid_size=grid_size, k=k)

    def init_states(self, batch_size: int, device: torch.device):
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        a = torch.zeros(batch_size, self.action_dim, device=device)
        return h, c, a

    def forward(self, x_t: torch.Tensor, states: tuple) -> tuple:
        h_prev, c_prev, a_prev = states

        # Step 1: Environment Observation — concat x_t, a_{t-1}, h_{t-1}
        obs = torch.cat([x_t, a_prev, h_prev], dim=-1)

        # Step 4 & 5: Internal State & Memory (LSTM logic with KAN layers)
        i_t = torch.sigmoid(self.kan_i(obs))
        f_t = torch.sigmoid(self.kan_f(obs))
        c_cand = torch.tanh(self.kan_c(obs))
        c_t = f_t * c_prev + i_t * c_cand
        o_t = torch.sigmoid(self.kan_o(obs))
        h_t = o_t * torch.tanh(c_t)

        # Step 6: Joint Agentic Heads
        a_cand = self.kan_policy(h_t)
        conf = torch.sigmoid(self.kan_critic(h_t))

        # Step 7: Internal Feedback & State Update
        # Modulates the candidate action using the Critic's confidence score
        a_t = a_cand * conf

        return a_t, (h_t, c_t, a_t)


class AgenticTKAN(nn.Module):
    """
    Improved AgenticTKAN:
    - FIX 1: All-timestep attention-weighted pooling (vs only using final hidden state).
    - FIX 4: KANLayer output projection head (vs flat nn.Linear).
    - FIX 5: Supports multi-layer cell stacking for depth-over-width approach.
    """
    def __init__(
        self,
        input_dim: int,
        horizon: int = 96,
        units: int = 32,       # FIX 5: reduced from 64 → 32
        grid_size: int = 5,
        k: int = 3,
        num_layers: int = 2,   # FIX 5: stacked depth
    ):
        super().__init__()
        self.horizon = horizon
        self.num_layers = num_layers
        self.units = units

        # FIX 5: Build a stack of cells (depth-over-width)
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input_dim = input_dim if i == 0 else units
            self.cells.append(AgenticTKANCell(cell_input_dim, units, grid_size=grid_size, k=k))

        # FIX 1: Attention weight layer for temporal pooling across all steps
        self.attn_w = nn.Linear(units, 1)

        # FIX 4: KAN projection layer instead of flat Linear
        self.fc = KANLayer(units, horizon, grid_size=grid_size, k=k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [Batch, Sequence_Len, Features]
        batch_size, seq_len, _ = x.shape

        # Unroll the RNN stack layer-by-layer
        layer_input = x  # [B, T, F]
        for layer_idx, cell in enumerate(self.cells):
            states = cell.init_states(batch_size, x.device)
            all_hidden = []
            for t in range(seq_len):
                a_t, states = cell(layer_input[:, t, :], states)
                all_hidden.append(states[0].unsqueeze(1))  # [B, 1, H]

            # Collect hidden states as sequence for next layer
            all_hidden = torch.cat(all_hidden, dim=1)  # [B, T, H]
            layer_input = all_hidden                   # pass to next cell layer

        # FIX 1: Attention-weighted pooling over all timesteps from the last layer
        attn_scores = self.attn_w(all_hidden)                           # [B, T, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)                # [B, T, 1]
        h_pooled = (all_hidden * attn_weights).sum(dim=1)               # [B, H]

        # FIX 4: KAN projection to output horizon
        out = self.fc(h_pooled)
        return out.unsqueeze(-1)  # [B, Horizon, 1]
