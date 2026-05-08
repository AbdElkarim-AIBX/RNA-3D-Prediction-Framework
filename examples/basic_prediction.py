"""
Example 1: Basic RNA Structure Prediction

Demonstrates the simplest usage of RNAFold-Net:
1. Define a sequence
2. Initialize model and trainer
3. Run optimization
4. Visualize result
"""

import torch
from rnafold_net import RNATransformer, sequence_to_numeric, visualize_rna_tensors
from rnafold_net.core import RNAFoldTrainer


def main():
    # Step 1: Define RNA sequence
    sequence = "AUCG"
    print(f"Target sequence: {sequence}")

    # Step 2: Convert to numeric tensor
    numeric_seq = sequence_to_numeric(sequence)
    num_nucs = len(sequence)

    # Step 3: Initialize Transformer model
    model = RNATransformer(
        vocab_size=5,      # 4 bases + padding
        d_model=128,       # Embedding dimension
        n_heads=8,         # Attention heads
        num_layers=4       # Encoder layers
    )

    # Step 4: Create training engine
    trainer = RNAFoldTrainer(
        model=model,
        num_nucs=num_nucs,
        total_epochs=1000,
        learning_rate=0.05,
        device='cpu'
    )

    # Step 5: Run optimization
    print("\nStarting optimization...")
    history = trainer.fit(numeric_seq, log_interval=100)

    # Step 6: Generate final structure
    backbone, full_structure = trainer.predict(numeric_seq)

    print(f"\nFinal structure:")
    print(f"  Backbone atoms: {backbone.shape[0]}")
    print(f"  Total atoms: {full_structure.shape[0]}")
    print(f"  Final loss: {history['total_loss'][-1]:.2f}")

    # Step 7: Visualize
    print("\nRendering 3D structure...")
    visualize_rna_tensors(
        full_structure,
        backbone_len=backbone.shape[0],
        num_nucs=num_nucs,
        filename="basic_prediction.html"
    )

    print("\nDone. Saved to basic_prediction.html")


if __name__ == "__main__":
    main()
