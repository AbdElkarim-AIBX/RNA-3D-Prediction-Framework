"""
End-to-End Training Engine for RNAFold-Net.

Integrates the Transformer model, NeRF geometry engine, physics loss
functions, and dynamic curriculum scheduler into a unified training
pipeline with full autograd support.

The training loop follows this sequence per epoch:
    1. AI predicts torsion angles from sequence
    2. NeRF builds 3D coordinates from angles
    3. Physics engine evaluates six energy terms
    4. Curriculum provides time-dependent weights
    5. Weighted combination forms master loss
    6. Backpropagation updates neural network parameters
"""

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

from rnafold_net.core import RNATransformer, build_rna_backbone, attach_rigid_bases
from rnafold_net.physics import (
    calculate_L_clash, calculate_L_Rg, calculate_L_coulomb,
    calculate_L_torsion, calculate_L_stacking, calculate_L_hbond,
    MasterCurriculum,
)


class RNAFoldTrainer:
    """
    Master training engine for RNA 3D structure prediction.

    Manages the complete training pipeline from sequence input to
    optimized 3D coordinates, with diagnostic telemetry and
    checkpointing support.

    Args:
        model (RNATransformer): Neural network model instance.
        num_nucs (int): Number of nucleotides in target sequence.
        total_epochs (int): Total training epochs. Default: 1000.
        learning_rate (float): Initial learning rate. Default: 0.05.
        lr_min (float): Minimum learning rate for cosine annealing.
            Default: 0.0001.
        grad_clip (float): Maximum gradient norm for clipping.
            Default: 2.0.
        device (str): Computation device ('cpu' or 'cuda').
            Default: 'cpu'.

    Attributes:
        model (RNATransformer): The neural network being trained.
        optimizer (torch.optim.Adam): Parameter optimizer.
        scheduler_lr (torch.optim.lr_scheduler.CosineAnnealingLR):
            Learning rate scheduler.
        curriculum (MasterCurriculum): Dynamic weight scheduler.
        history (dict): Training metrics logged per epoch.

    Example:
        >>> model = RNATransformer()
        >>> trainer = RNAFoldTrainer(model, num_nucs=4)
        >>> trainer.fit(numeric_sequence)
    """

    def __init__(self, model, num_nucs, total_epochs=1000,
                 learning_rate=0.05, lr_min=0.0001, grad_clip=2.0,
                 device='cpu'):
        self.model = model.to(device)
        self.num_nucs = num_nucs
        self.total_epochs = total_epochs
        self.grad_clip = grad_clip
        self.device = device

        # Optimizer with high initial learning rate for exploration
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        # Cosine annealing simulates slow cooling (simulated annealing)
        self.scheduler_lr = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs,
            eta_min=lr_min
        )

        # Dynamic curriculum for phased constraint activation
        self.curriculum = MasterCurriculum(total_epochs=total_epochs)

        # Training history for analysis
        self.history = {
            'total_loss': [],
            'clash': [],
            'rg': [],
            'coulomb': [],
            'torsion': [],
            'stacking': [],
            'hbond': [],
            'learning_rate': [],
        }

    def compute_physics_loss(self, predicted_angles, rna_backbone, full_molecule,
                             bases_coords, epoch):
        """
        Computes weighted combination of six physics loss terms.

        Args:
            predicted_angles (torch.Tensor): Predicted torsion angles.
            rna_backbone (torch.Tensor): Backbone coordinates.
            full_molecule (torch.Tensor): Complete molecule coordinates.
            bases_coords (torch.Tensor): Base-only coordinates.
            epoch (int): Current training epoch.

        Returns:
            tuple: (total_loss, individual_losses_dict)
        """
        # Calculate all six physical forces
        loss_clash = calculate_L_clash(full_molecule, min_distance=2.5)
        loss_rg = calculate_L_Rg(full_molecule)
        loss_coulomb = calculate_L_coulomb(rna_backbone)
        loss_torsion = calculate_L_torsion(predicted_angles)
        loss_stacking = calculate_L_stacking(bases_coords)
        loss_hbond = calculate_L_hbond(bases_coords)

        # Get curriculum weights for this epoch
        w_clash, w_rg, w_hbond, w_coulomb, w_torsion, w_stack =             self.curriculum.get_weights(epoch)

        # Weighted master loss function
        total_loss = (
            w_clash * loss_clash +
            w_rg * loss_rg +
            w_coulomb * loss_coulomb +
            w_torsion * loss_torsion +
            w_stack * loss_stacking +
            w_hbond * loss_hbond
        )

        losses = {
            'clash': (w_clash * loss_clash).item(),
            'rg': (w_rg * loss_rg).item(),
            'coulomb': (w_coulomb * loss_coulomb).item(),
            'torsion': (w_torsion * loss_torsion).item(),
            'stacking': (w_stack * loss_stacking).item(),
            'hbond': (w_hbond * loss_hbond).item(),
            'raw_clash': loss_clash.item(),
            'raw_rg': loss_rg.item(),
            'raw_hbond': loss_hbond.item(),
        }

        return total_loss, losses

    def fit(self, numeric_sequence, log_interval=100):
        """
        Runs the complete training loop.

        Args:
            numeric_sequence (torch.Tensor): Input sequence tensor,
                shape [1, seq_len].
            log_interval (int): Epochs between logging. Default: 100.

        Returns:
            dict: Final training history with all logged metrics.

        Note:
            The sequence tensor should be on the same device as the model.
        """
        numeric_sequence = numeric_sequence.to(self.device)

        print("Starting RNAFold-Net Training")
        print(f"Sequence length: {self.num_nucs} nucleotides")
        print(f"Total epochs: {self.total_epochs}")
        print(f"Device: {self.device}")
        print("-" * 50)

        for epoch in range(self.total_epochs):
            self.optimizer.zero_grad()

            # Forward Pass 1: AI predicts angles from sequence
            predicted_angles = self.model(numeric_sequence)[0]
            ai_backbone_angles = predicted_angles[:, :6]
            ai_chi_angles = predicted_angles[:, 6]

            # Forward Pass 2: NeRF builds 3D geometry
            rna_backbone = build_rna_backbone(
                ai_backbone_angles, self.num_nucs
            )
            full_molecule = attach_rigid_bases(
                rna_backbone, self.num_nucs, ai_chi_angles
            )

            # Separate bases for H-bond and stacking calculations
            bases_coords = full_molecule[rna_backbone.shape[0]:]

            # Forward Pass 3: Physics engine evaluates structure
            total_loss, losses = self.compute_physics_loss(
                predicted_angles, rna_backbone,
                full_molecule, bases_coords, epoch
            )

            # Backward Pass: gradients flow through entire pipeline
            total_loss.backward()

            # Gradient clipping prevents instability from physics spikes
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip
            )

            self.optimizer.step()
            self.scheduler_lr.step()

            # Log metrics
            current_lr = self.scheduler_lr.get_last_lr()[0]
            self.history['total_loss'].append(total_loss.item())
            self.history['clash'].append(losses['clash'])
            self.history['rg'].append(losses['rg'])
            self.history['coulomb'].append(losses['coulomb'])
            self.history['torsion'].append(losses['torsion'])
            self.history['stacking'].append(losses['stacking'])
            self.history['hbond'].append(losses['hbond'])
            self.history['learning_rate'].append(current_lr)

            # Diagnostic telemetry
            if epoch % log_interval == 0 or epoch == self.total_epochs - 1:
                print(f"
Epoch {epoch:4d} | LR: {current_lr:.6f}")
                print(f"  Total Energy : {total_loss.item():9.2f}")
                print(f"  1. Clash     : {losses['clash']:9.2f} (raw: {losses['raw_clash']:.2f})")
                print(f"  2. Compact   : {losses['rg']:9.2f} (raw: {losses['raw_rg']:.2f})")
                print(f"  3. Coulomb   : {losses['coulomb']:9.2f}")
                print(f"  4. Torsion   : {losses['torsion']:9.2f}")
                print(f"  5. Stacking  : {losses['stacking']:9.2f}")
                print(f"  6. H-Bond    : {losses['hbond']:9.2f} (raw: {losses['raw_hbond']:.2f})")

                # Stage transition notifications
                if epoch == int(0.5 * self.total_epochs):
                    print("   >> Stage 3: Torsion and Stacking activated")
                if epoch == int(0.7 * self.total_epochs):
                    print("   >> Stage 4: Biological base-pairing activated")

        print("
" + "-" * 50)
        print("Training complete.")
        return self.history

    def predict(self, numeric_sequence):
        """
        Generates 3D structure prediction for a given sequence.

        Args:
            numeric_sequence (torch.Tensor): Input sequence,
                shape [1, seq_len].

        Returns:
            tuple: (backbone_coords, full_molecule_coords)
                backbone_coords: torch.Tensor [num_backbone_atoms, 3]
                full_molecule_coords: torch.Tensor [total_atoms, 3]
        """
        self.model.eval()
        numeric_sequence = numeric_sequence.to(self.device)

        with torch.no_grad():
            predicted_angles = self.model(numeric_sequence)[0]
            ai_backbone_angles = predicted_angles[:, :6]
            ai_chi_angles = predicted_angles[:, 6]

            rna_backbone = build_rna_backbone(
                ai_backbone_angles, self.num_nucs
            )
            full_molecule = attach_rigid_bases(
                rna_backbone, self.num_nucs, ai_chi_angles
            )

        return rna_backbone, full_molecule
