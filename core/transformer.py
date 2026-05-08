"""
Transformer-Based Sequence Encoder for RNA Structure Prediction.

Processes nucleotide sequences through multi-head self-attention layers
to capture long-range base-pairing dependencies, then projects learned
representations into seven torsion angles per nucleotide.

Architecture:
    - Nucleotide embedding (4 bases + padding -> d_model dimensions)
    - Positional encoding (learnable position embeddings)
    - Transformer encoder (multi-head self-attention stack)
    - MLP prediction head (hidden state -> 7 angles)

Reference:
    Vaswani et al. (2017) "Attention is all you need"
    NeurIPS 30.
"""

import torch
import torch.nn as nn
import numpy as np


class RNATransformer(nn.Module):
    """
    Transformer-based RNA structure prediction model.

    Maps nucleotide sequences to torsion angle predictions through
    multi-head self-attention and MLP projection.

    Args:
        vocab_size (int): Number of distinct tokens (4 bases + padding).
            Default: 5.
        d_model (int): Embedding dimensionality. Default: 128.
        n_heads (int): Number of attention heads. Default: 8.
        num_layers (int): Transformer encoder layers. Default: 4.
        max_seq_len (int): Maximum sequence length. Default: 200.
        output_angles (int): Number of output angles per nucleotide.
            Default: 7 (6 backbone + 1 chi).

    Attributes:
        nucleotide_embedding (nn.Embedding): Maps token indices to vectors.
        position_embedding (nn.Embedding): Learnable position encodings.
        transformer (nn.TransformerEncoder): Self-attention stack.
        angle_predictor (nn.Sequential): MLP head for angle regression.

    Example:
        >>> model = RNATransformer(vocab_size=5, d_model=128)
        >>> sequence = torch.tensor([[0, 1, 2, 3]])  # AUCG
        >>> angles = model(sequence)
        >>> print(angles.shape)  # [1, 4, 7]
    """

    def __init__(self, vocab_size=5, d_model=128, n_heads=8,
                 num_layers=4, max_seq_len=200, output_angles=7):
        super().__init__()

        # Nucleotide embedding: token index -> continuous vector
        self.nucleotide_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding: sequence position -> continuous vector
        # Learnable embeddings allow the model to adapt position information
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder: multi-head self-attention stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=512,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # MLP prediction head: hidden state -> torsion angles
        # Two-layer perceptron with ReLU activation
        self.angle_predictor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, output_angles)
        )

    def forward(self, x):
        """
        Forward pass: sequence -> angle predictions.

        Args:
            x (torch.Tensor): Integer sequence indices,
                shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Predicted angles in range [0, 2*pi],
                shape [batch_size, seq_len, output_angles].
        """
        seq_length = x.size(1)

        # Generate position IDs (0, 1, 2, ...)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)

        # Combine nucleotide identity with positional information
        x_emb = self.nucleotide_embedding(x) + self.position_embedding(positions)

        # Process through self-attention layers
        memory = self.transformer(x_emb)

        # Predict raw angle values
        raw_angles = self.angle_predictor(memory)

        # Scale to [0, 2*pi] range using sigmoid
        predicted_angles = torch.sigmoid(raw_angles) * 2 * np.pi

        return predicted_angles
