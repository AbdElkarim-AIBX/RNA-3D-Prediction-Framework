"""
Dynamic Curriculum Weight Scheduler for Phased Training.

Implements a temporal phasing strategy inspired by simulated annealing,
where geometric constraints are introduced early and biological constraints
activate during late-stage refinement.

Scheduling Strategy:
    Stage 1 (Epochs 0-200): Geometric constraints (clash, Coulomb)
    Stage 2 (Epochs 200-500): Compaction ramps up via sigmoid warmup
    Stage 3 (Epochs 500-700): Torsion and stacking activate
    Stage 4 (Epochs 700-1000): Hydrogen bonds lock in secondary structure

This staged activation mirrors natural RNA folding where secondary structure
nucleates early and tertiary contacts lock in subsequently.
"""

import numpy as np


class MasterCurriculum:
    """
    Dynamic weight scheduler implementing phased training protocol.

    Controls the relative weighting of six physics loss terms across
    training epochs to prevent conflicting forces from creating
    adversarial gradient landscapes.

    Args:
        total_epochs (int): Total number of training epochs.
            Default: 1000.

    Attributes:
        total (int): Stored total epoch count for normalization.

    Methods:
        get_weights(epoch): Returns tuple of weights for current epoch.
        get_lambda_clash(epoch): Exponential decay for clash weight.
        get_lambda_rg(epoch): Sigmoid warmup for compaction weight.
        get_lambda_hbond(epoch): Late-stage activation for H-bonds.
        get_lambda_stack(epoch): Mid-stage activation for stacking.
        get_lambda_torsion(epoch): Mid-stage activation for torsion.

    Example:
        >>> scheduler = MasterCurriculum(total_epochs=1000)
        >>> w_clash, w_rg, w_hbond, w_coulomb, w_torsion, w_stack =         ...     scheduler.get_weights(500)
    """

    def __init__(self, total_epochs=1000):
        self.total = total_epochs

    def get_lambda_clash(self, current_epoch, lambda_max=100.0, decay_rate=5.0):
        """
        Exponential decay: starts high to untangle, drops off.

        The steric clash weight follows exponential decay from a high
        initial value, providing strong repulsive force during early
        training that rapidly resolves atomic overlaps.

        Args:
            current_epoch (int): Current training epoch.
            lambda_max (float): Initial clash weight. Default: 100.0.
            decay_rate (float): Exponential decay rate. Default: 5.0.

        Returns:
            float: Clash weight for current epoch.
        """
        t = current_epoch / self.total
        return lambda_max * np.exp(-decay_rate * t)

    def get_lambda_rg(self, current_epoch, lambda_final=10.0,
                      steepness=15.0, midpoint=0.4):
        """
        Sigmoid warmup: starts at 0, ramps up in the middle.

        The radius of gyration weight uses sigmoid warmup that ramps up
        during mid-training, preventing premature compaction before local
        steric conflicts are resolved.

        Args:
            current_epoch (int): Current training epoch.
            lambda_final (float): Final compaction weight. Default: 10.0.
            steepness (float): Sigmoid steepness parameter. Default: 15.0.
            midpoint (float): Normalized midpoint of activation. Default: 0.4.

        Returns:
            float: Compaction weight for current epoch.
        """
        t = current_epoch / self.total
        return lambda_final / (1.0 + np.exp(-steepness * (t - midpoint)))

    def get_lambda_hbond(self, current_epoch, lambda_max=50.0,
                         steepness=50.0, activation_point=0.7):
        """
        Late-stage activation: turns on sharply at specified epoch.

        Hydrogen bonding activates only in the final portion of training,
        preventing premature formation of incorrect base pairs that would
        trap the structure in non-native topologies.

        Args:
            current_epoch (int): Current training epoch.
            lambda_max (float): Maximum H-bond weight. Default: 50.0.
            steepness (float): Sigmoid steepness. Default: 50.0.
            activation_point (float): Normalized activation threshold.
                Default: 0.7 (epoch 700 of 1000).

        Returns:
            float: Hydrogen bond weight for current epoch.
        """
        t = current_epoch / self.total
        return lambda_max / (1.0 + np.exp(-steepness * (t - activation_point)))

    def get_lambda_stack(self, current_epoch, lambda_max=5.0,
                         steepness=40.0, activation_point=0.5):
        """
        Mid-stage activation for base stacking forces.

        Pi-pi stacking interactions activate around the halfway point,
        guiding base orientation after global compaction but before
        specific hydrogen bond formation.

        Args:
            current_epoch (int): Current training epoch.
            lambda_max (float): Maximum stacking weight. Default: 5.0.
            steepness (float): Sigmoid steepness. Default: 40.0.
            activation_point (float): Normalized activation threshold.
                Default: 0.5 (epoch 500 of 1000).

        Returns:
            float: Stacking weight for current epoch.
        """
        t = current_epoch / self.total
        return lambda_max / (1.0 + np.exp(-steepness * (t - activation_point)))

    def get_lambda_torsion(self, current_epoch, lambda_max=2.0,
                           steepness=40.0, activation_point=0.5):
        """
        Mid-stage activation for torsional strain constraints.

        Torsional preferences activate alongside stacking to guide
        backbone rotameric states during the refinement phase.

        Args:
            current_epoch (int): Current training epoch.
            lambda_max (float): Maximum torsion weight. Default: 2.0.
            steepness (float): Sigmoid steepness. Default: 40.0.
            activation_point (float): Normalized activation threshold.
                Default: 0.5.

        Returns:
            float: Torsion weight for current epoch.
        """
        t = current_epoch / self.total
        return lambda_max / (1.0 + np.exp(-steepness * (t - activation_point)))

    def get_weights(self, epoch):
        """
        Returns all six curriculum weights for the specified epoch.

        This is the primary interface for the training loop.

        Args:
            epoch (int): Current training epoch.

        Returns:
            tuple: (w_clash, w_rg, w_hbond, w_coulomb, w_torsion, w_stack)
                All weights are floats in the range [0, lambda_max].

        Note:
            Coulomb weight is constant at 1.0 (always active, water-screened).
            The order matches the MasterLoss computation in training loops.
        """
        t = epoch / self.total

        # Stage 1: Geometric constraints (always active)
        w_clash = self.get_lambda_clash(epoch)
        w_coulomb = 1.0  # Always neutralized by water

        # Stage 2: Compaction slowly ramps up
        w_rg = self.get_lambda_rg(epoch)

        # Stage 3: Torsion and stacking turn on at midpoint
        w_torsion = self.get_lambda_torsion(epoch)
        w_stack = self.get_lambda_stack(epoch)

        # Stage 4: Biology locks in during final phase
        w_hbond = self.get_lambda_hbond(epoch)

        return w_clash, w_rg, w_hbond, w_coulomb, w_torsion, w_stack
