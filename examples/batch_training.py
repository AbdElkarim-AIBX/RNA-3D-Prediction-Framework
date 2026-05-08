"""
Example 3: Batch Training with Synthetic Supervised Data

Demonstrates training the model on a dataset with ground-truth
coordinates, combining supervised MSE loss with physics constraints.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rnafold_net import RNATransformer
from rnafold_net.data import RNADataset
from rnafold_net.physics import calculate_L_clash


def main():
    # Create synthetic dataset
    dataset = RNADataset(num_samples=100, seq_len=10, atoms_per_nuc=9)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize model
    model = RNATransformer(
        vocab_size=5,
        d_model=128,
        n_heads=8,
        num_layers=4,
        output_angles=9  # 9 outputs for direct XYZ supervision
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training on synthetic supervised data...")

    for epoch in range(20):
        epoch_loss = 0.0

        for batch_seq, batch_truth in loader:
            optimizer.zero_grad()

            # Forward pass
            preds = model(batch_seq)  # [batch, seq_len, 9]

            # Supervised loss: MSE against ground truth
            supervised_loss = F.mse_loss(preds, batch_truth)

            # Physics guardrail: steric clash penalty
            coords_3d = preds.view(batch_seq.shape[0], -1, 3)
            dist_matrix = torch.cdist(coords_3d, coords_3d)
            # Ignore self-distances
            dist_matrix = dist_matrix + torch.eye(
                dist_matrix.size(-1), device=preds.device
            ) * 10
            clash_penalty = torch.relu(1.2 - dist_matrix).sum()

            # Combined loss (80% data, 20% physics)
            master_loss = (0.8 * supervised_loss) + (0.2 * clash_penalty)

            master_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += master_loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1:02d}/20 | Loss: {avg_loss:.4f}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
