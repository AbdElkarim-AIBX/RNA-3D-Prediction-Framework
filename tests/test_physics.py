"""
Tests for physics-informed loss functions.
"""

import torch
import pytest
import numpy as np

from rnafold_net.physics import (
    calculate_L_clash, calculate_L_Rg, calculate_L_coulomb,
    calculate_L_torsion, calculate_L_stacking, calculate_L_hbond,
    MasterCurriculum,
)


class TestPhysicsLosses:
    """Test suite for physics engine components."""

    def test_clash_no_violation(self):
        """Clash loss should be zero for well-separated atoms."""
        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0]
        ])
        loss = calculate_L_clash(coords, min_distance=1.5)
        assert loss.item() == 0.0

    def test_clash_detects_overlap(self):
        """Clash loss should be positive for overlapping atoms."""
        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0]
        ])
        loss = calculate_L_clash(coords, min_distance=1.5)
        assert loss.item() > 0.0

    def test_rg_decreases_with_compaction(self):
        """Radius of gyration should decrease for compact structures."""
        spread_coords = torch.randn(50, 3) * 10  # Spread out
        compact_coords = torch.randn(50, 3) * 2    # Compact

        rg_spread = calculate_L_Rg(spread_coords)
        rg_compact = calculate_L_Rg(compact_coords)

        assert rg_spread > rg_compact

    def test_coulomb_vacuum_vs_water(self):
        """Vacuum should have higher repulsion than water."""
        coords = torch.randn(12, 3)  # 2 phosphates

        vacuum = calculate_L_coulomb(coords, epsilon_r=1.0)
        water = calculate_L_coulomb(coords, epsilon_r=80.0)

        assert vacuum > water

    def test_torsion_minimum_at_staggered(self):
        """Torsional energy should be lower at staggered conformations."""
        staggered = torch.ones(1) * (np.pi / 3)  # 60 degrees
        eclipsed = torch.ones(1) * 0.0           # 0 degrees

        loss_staggered = calculate_L_torsion(staggered)
        loss_eclipsed = calculate_L_torsion(eclipsed)

        assert loss_staggered < loss_eclipsed

    def test_stacking_well_minimum(self):
        """Stacking energy should be most negative at ideal distance."""
        # Two bases at exactly 3.4 Angstroms
        perfect = torch.tensor([[0., 0., 0.], [0., 0., 3.4]]).repeat_interleave(5, dim=0)
        # Two bases crashing at 2.0 Angstroms
        clash = torch.tensor([[0., 0., 0.], [0., 0., 2.0]]).repeat_interleave(5, dim=0)

        energy_perfect = calculate_L_stacking(perfect)
        energy_clash = calculate_L_stacking(clash)

        # Perfect should be more negative (stronger attraction)
        assert energy_perfect < energy_clash

    def test_hbond_rewards_proximity(self):
        """H-bond loss should decrease as bases approach ideal distance."""
        far = torch.tensor([[0., 0., 0.], [0., 0., 10.0]]).repeat_interleave(5, dim=0)
        close = torch.tensor([[0., 0., 0.], [0., 0., 3.0]]).repeat_interleave(5, dim=0)

        loss_far = calculate_L_hbond(far, pairs=[(0, 1)])
        loss_close = calculate_L_hbond(close, pairs=[(0, 1)])

        assert loss_far > loss_close


class TestCurriculum:
    """Test suite for dynamic curriculum scheduler."""

    def test_clash_decay(self):
        """Clash weight should decrease over epochs."""
        scheduler = MasterCurriculum(total_epochs=1000)

        early = scheduler.get_lambda_clash(0)
        late = scheduler.get_lambda_clash(999)

        assert early > late
        assert early > 0
        assert late >= 0

    def test_rg_warmup(self):
        """Rg weight should increase over epochs."""
        scheduler = MasterCurriculum(total_epochs=1000)

        early = scheduler.get_lambda_rg(0)
        late = scheduler.get_lambda_rg(999)

        assert late > early
        assert early >= 0

    def test_hbond_late_activation(self):
        """H-bond should be near zero early and high late."""
        scheduler = MasterCurriculum(total_epochs=1000)

        early = scheduler.get_lambda_hbond(100)
        mid = scheduler.get_lambda_hbond(500)
        late = scheduler.get_lambda_hbond(900)

        assert early < 1.0
        assert late > 10.0
        assert mid < late
