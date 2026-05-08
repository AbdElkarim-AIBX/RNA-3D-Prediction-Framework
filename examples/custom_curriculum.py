"""
Example 2: Custom Curriculum Configuration

Shows how to modify the dynamic curriculum weights for different
training scenarios or molecular sizes.
"""

import torch
from rnafold_net import RNATransformer, sequence_to_numeric, visualize_rna_tensors
from rnafold_net.core import RNAFoldTrainer
from rnafold_net.physics import MasterCurriculum


class CustomCurriculum(MasterCurriculum):
    """
    Custom curriculum for longer sequences requiring extended clash resolution.
    """

    def __init__(self, total_epochs=1000):
        super().__init__(total_epochs)

    def get_lambda_clash(self, current_epoch, lambda_max=150.0, decay_rate=3.0):
        # Slower decay for longer untangling phase
        t = current_epoch / self.total
        return lambda_max * np.exp(-decay_rate * t)

    def get_lambda_rg(self, current_epoch, lambda_final=15.0,
                      steepness=10.0, midpoint=0.3):
        # Earlier compaction for faster folding
        t = current_epoch / self.total
        return lambda_final / (1.0 + np.exp(-steepness * (t - midpoint)))


def main():
    sequence = "AUCGAUCGAUCG"  # Longer 12-nt sequence
    numeric_seq = sequence_to_numeric(sequence)
    num_nucs = len(sequence)

    model = RNATransformer(d_model=128, n_heads=8, num_layers=4)

    # Use custom curriculum
    trainer = RNAFoldTrainer(
        model=model,
        num_nucs=num_nucs,
        total_epochs=1000,
        learning_rate=0.05
    )
    # Override with custom scheduler
    trainer.curriculum = CustomCurriculum(total_epochs=1000)

    print(f"Training with custom curriculum on {num_nucs}-nt sequence")
    history = trainer.fit(numeric_seq, log_interval=100)

    backbone, full_structure = trainer.predict(numeric_seq)
    visualize_rna_tensors(
        full_structure,
        backbone_len=backbone.shape[0],
        num_nucs=num_nucs,
        filename="custom_curriculum.html"
    )

    print("Done. Saved to custom_curriculum.html")


if __name__ == "__main__":
    main()
