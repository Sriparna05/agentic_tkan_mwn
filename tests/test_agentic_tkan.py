import torch
import pytest
from src.models.agentic_kan.tkan_agentic import AgenticTKAN, AgenticTKANCell
from src.models.kan.tkan import TKAN, TKANCell

def test_agentic_tkan_cell_initialization_and_states():
    input_dim = 4
    hidden_dim = 32
    cell = AgenticTKANCell(input_dim=input_dim, hidden_dim=hidden_dim, grid_size=5, k=3)
    
    # State shapes
    batch_size = 4
    h, c, a = cell.init_states(batch_size, torch.device("cpu"))
    assert h.shape == (batch_size, hidden_dim)
    assert c.shape == (batch_size, hidden_dim)
    assert a.shape == (batch_size, 1)
    assert (h == 0).all()
    assert (c == 0).all()
    assert (a == 0).all()
    
    # Forward step
    x_t = torch.randn(batch_size, input_dim)
    a_t, (h_next, c_next, a_next) = cell(x_t, (h, c, a))
    assert a_t.shape == (batch_size, 1)
    assert h_next.shape == (batch_size, hidden_dim)
    assert c_next.shape == (batch_size, hidden_dim)
    assert a_next.shape == (batch_size, 1)

def test_agentic_tkan_full_forward_and_backward():
    batch_size = 4
    seq_len = 16
    in_features = 4
    units = 32
    
    model = AgenticTKAN(input_dim=in_features, horizon=1, units=units, num_layers=1)
    x = torch.randn(batch_size, seq_len, in_features)
    
    out = model(x).squeeze(-1) # [batch_size, 1]
    if out.dim() == 2 and out.size(-1) == 1:
        out = out.squeeze(-1)
    assert out.shape == (batch_size,)
    
    # Ensure all parameters receive gradients
    loss = out.sum()
    loss.backward()
    
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient!"

def test_agentic_tkan_parameter_registration():
    model = AgenticTKAN(input_dim=4, horizon=1, units=32, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Verify optimizer contains all parameter groups before training
    opt_params = sum(len(group['params']) for group in optimizer.param_groups)
    model_params = len(list(model.parameters()))
    assert opt_params == model_params

def test_standard_tkan_forward_and_backward():
    batch_size = 4
    seq_len = 16
    in_features = 4
    units = 32
    
    model = TKAN(input_dim=in_features, horizon=1, units=units, num_sub_kans=1)
    x = torch.randn(batch_size, seq_len, in_features)
    
    out = model(x).squeeze(-1)
    if out.dim() == 2 and out.size(-1) == 1:
        out = out.squeeze(-1)
    assert out.shape == (batch_size,)
    
    loss = out.sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"TKAN param {name} has no gradient!"
