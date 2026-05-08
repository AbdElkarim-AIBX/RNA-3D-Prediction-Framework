"""
Tests for core NeRF geometry and backbone construction.
"""

import torch
import pytest
import numpy as np

from rnafold_net.core import calculate_nerf, build_rna_backbone


class TestNeRF:
    """Test suite for differentiable NeRF algorithm."""

    def test_gradient_flow(self):
        """Verify gradients flow from coordinates back to torsion angle."""
        atom_A = torch.tensor([0.0, 0.0, 0.0])
        atom_B = torch.tensor([1.5, 0.0, 0.0])
        atom_C = torch.tensor([2.0, 1.2, 0.0])

        bond_length = torch.tensor(1.54)
        bond_angle = torch.tensor(1.91)

        # Predicted torsion with gradient tracking
        predicted_torsion = torch.tensor(np.pi / 3, requires_grad=True)

        atom_D = calculate_nerf(atom_A, atom_B, atom_C, bond_length, bond_angle, predicted_torsion)

        # Simple loss: distance from origin
        loss = torch.sum(atom_D ** 2)
        loss.backward()

        assert predicted_torsion.grad is not None
        assert predicted_torsion.grad.item() != 0.0

    def test_backbone_construction(self):
        """Test RNA backbone builds correct number of atoms."""
        num_nucs = 4
        angles = torch.nn.Parameter(torch.rand(num_nucs, 6) * 2 * np.pi)

        backbone = build_rna_backbone(angles, num_nucs)

        # 3 anchor atoms + 6 atoms per nucleotide
        expected_atoms = 3 + (num_nucs * 6)
        assert backbone.shape[0] == expected_atoms
        assert backbone.shape[1] == 3  # XYZ coordinates

    def test_differentiability(self):
        """Ensure entire backbone construction is differentiable."""
        num_nucs = 2
        angles = torch.nn.Parameter(torch.rand(num_nucs, 6) * 2 * np.pi)

        backbone = build_rna_backbone(angles, num_nucs)
        loss = backbone.sum()
        loss.backward()

        assert angles.grad is not None
        assert angles.grad.shape == angles.shape
