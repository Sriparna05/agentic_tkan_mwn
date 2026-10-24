import torch
import numpy as np
import pytest
from src.models.deep_learning.mlp import SmallMLP
from src.models.deep_learning.gru import SingleLayerGRU

def test_small_mlp_forward_and_params():
    seq_len = 16
    in_features = 4
    hidden_dim = 32
    model = SmallMLP(seq_len=seq_len, in_features=in_features, hidden_dim=hidden_dim, out_features=1)
    
    # Param count: (64 * 32 + 32) + (32 * 1 + 1) = 2080 + 33 = 2113
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == (16 * 4 * 32 + 32) + (32 * 1 + 1)
    
    batch_size = 8
    x = torch.randn(batch_size, seq_len, in_features)
    out = model(x)
    assert out.shape == (batch_size,)
    
    # Test backward pass
    loss = out.sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None

def test_single_layer_gru_forward_and_params():
    seq_len = 16
    in_features = 4
    hidden_dim = 32
    model = SingleLayerGRU(in_features=in_features, hidden_dim=hidden_dim, num_layers=1)
    
    # GRU params: 3 * (input_size * hidden + hidden * hidden + 2 * hidden) + (hidden * 1 + 1)
    # = 3 * (4*32 + 32*32 + 64) + 33 = 3 * (128 + 1024 + 64) + 33 = 3 * 1216 + 33 = 3681
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 3681
    
    batch_size = 8
    x = torch.randn(batch_size, seq_len, in_features)
    out = model(x)
    assert out.shape == (batch_size,)
    
    # Test backward pass
    loss = out.sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None

def test_persistence_logic():
    batch_size = 10
    seq_len = 16
    n_features = 4
    # Feature 0 is dl_buffer
    raw_x = np.random.uniform(0, 10000, size=(batch_size, seq_len, n_features)).astype(np.float32)
    # Latest available is index 15
    pred_persistence = raw_x[:, 15, 0]
    assert pred_persistence.shape == (batch_size,)
    np.testing.assert_array_equal(pred_persistence, raw_x[:, -1, 0])

def test_inverse_transform():
    y_true_bytes = np.array([0.0, 10.0, 1000.0, 50000.0], dtype=np.float32)
    y_log1p = np.log1p(y_true_bytes)
    y_recov = np.expm1(y_log1p)
    np.testing.assert_allclose(y_true_bytes, y_recov, rtol=1e-5)
