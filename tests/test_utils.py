"""
Tests for utility functions.
"""

import torch
import pytest

from rnafold_net.utils import sequence_to_numeric, numeric_to_sequence, validate_sequence


class TestSequenceProcessing:
    """Test suite for sequence encoding/decoding."""

    def test_sequence_to_numeric(self):
        """Convert string sequence to numeric tensor."""
        result = sequence_to_numeric("AUCG")
        expected = torch.tensor([[0, 3, 1, 2]])
        assert torch.equal(result, expected)

    def test_numeric_to_sequence(self):
        """Convert numeric indices back to string."""
        result = numeric_to_sequence([0, 3, 1, 2])
        assert result == "AUCG"

    def test_invalid_character_raises(self):
        """Invalid characters should raise ValueError."""
        with pytest.raises(ValueError):
            sequence_to_numeric("AUCGX")

    def test_validate_sequence(self):
        """Sequence validation should detect invalid characters."""
        assert validate_sequence("AUCG") is True
        assert validate_sequence("AUCGX") is False

    def test_case_insensitive(self):
        """Lowercase input should be handled."""
        result = sequence_to_numeric("aucg")
        expected = torch.tensor([[0, 3, 1, 2]])
        assert torch.equal(result, expected)
