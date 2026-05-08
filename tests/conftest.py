"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch


@pytest.fixture
def sample_sequence():
    """Standard test sequence."""
    return torch.tensor([[0, 3, 1, 2]], dtype=torch.long)  # AUCG


@pytest.fixture
def small_model():
    """Small model for fast testing."""
    from rnafold_net import RNATransformer
    return RNATransformer(vocab_size=5, d_model=32, n_heads=4, num_layers=2)
