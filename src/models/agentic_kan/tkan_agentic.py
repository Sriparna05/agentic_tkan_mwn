import torch
import torch.nn as nn
try:
    from src.models.kan.vanilla_kan import KANLayer
except ImportError:
    from models.kan.vanilla_kan import KANLayer



class AgenticTKANCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, grid_size: int = 5, k: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = 1
        concat_dim = input_dim + self.action_dim + hidden_dim

        # Fused KAN-based LSTM gates: i, f, c, o (exact same 4x hidden_dim capacity)
        self.kan_gates = KANLayer(concat_dim, 4 * hidden_dim, grid_size=grid_size, k=k)

        # Fused Joint Agentic Heads: Branch A (Policy) + Branch B (Critic)
        self.kan_heads = KANLayer(hidden_dim, 2 * self.action_dim, grid_size=grid_size, k=k)

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
        gates = self.kan_gates(obs)
        i_raw, f_raw, c_raw, o_raw = torch.split(gates, self.hidden_dim, dim=-1)
        i_t = torch.sigmoid(i_raw)
        f_t = torch.sigmoid(f_raw)
        c_cand = torch.tanh(c_raw)
        c_t = f_t * c_prev + i_t * c_cand
        o_t = torch.sigmoid(o_raw)
        h_t = o_t * torch.tanh(c_t)

        # Step 6: Joint Agentic Heads
        heads = self.kan_heads(h_t)
        a_cand, critic_raw = torch.split(heads, self.action_dim, dim=-1)
        conf = torch.sigmoid(critic_raw)

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
