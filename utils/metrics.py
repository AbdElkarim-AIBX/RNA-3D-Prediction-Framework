"""
Structural Accuracy Metrics for RNA 3D Structure Comparison.

Implements RMSD (Root Mean Square Deviation) calculation with optional
Kabsch algorithm for optimal rigid-body alignment before comparison.

The Kabsch algorithm computes the optimal rotation matrix that minimizes
RMSD between two point sets, removing arbitrary coordinate frame bias.
"""

import torch
import torch.nn.functional as F


def calculate_rmsd(pred, target):
    """
    Calculates standard RMSD between predicted and target coordinates.

    Computes the root mean square deviation without alignment.
    Use calculate_kabsch_rmsd for scientific comparisons that require
    optimal superposition.

    Args:
        pred (torch.Tensor): Predicted coordinates, shape [..., 3].
        target (torch.Tensor): Target coordinates, shape [..., 3].

    Returns:
        torch.Tensor: RMSD value in Angstroms (scalar).

    Example:
        >>> pred = torch.randn(100, 3)
        >>> target = torch.randn(100, 3)
        >>> rmsd = calculate_rmsd(pred, target)
    """
    diff = pred - target
    mse = torch.mean(torch.sum(diff ** 2, dim=-1))
    return torch.sqrt(mse)


def calculate_kabsch_rmsd(pred, target):
    """
    Calculates RMSD after optimal alignment via Kabsch algorithm.

    The Kabsch algorithm finds the optimal rotation matrix that minimizes
    the RMSD between two sets of points. This is the standard scientific
    metric for structure comparison, as it removes arbitrary coordinate
    frame choices.

    Algorithm steps:
        1. Center both structures at origin (translation invariance)
        2. Compute cross-covariance matrix
        3. Singular Value Decomposition (SVD) for optimal rotation
        4. Apply rotation and measure deviation

    Args:
        pred (torch.Tensor): Predicted coordinates, shape [batch, atoms, 3]
            or [atoms, 3].
        target (torch.Tensor): Target coordinates, same shape as pred.

    Returns:
        torch.Tensor: Kabsch-aligned RMSD in Angstroms (scalar).

    Reference:
        Kabsch, W. (1976). A solution for the best rotation to relate
        two sets of vectors. Acta Crystallographica A32, 922-923.

    Example:
        >>> pred = torch.randn(1, 100, 3)
        >>> target = torch.randn(1, 100, 3)
        >>> rmsd = calculate_kabsch_rmsd(pred, target)
    """
    # Reshape to [Batch, Atoms, 3] if necessary
    p = pred.view(pred.size(0), -1, 3)
    t = target.view(target.size(0), -1, 3)

    # Center at origin
    p_centered = p - p.mean(dim=1, keepdim=True)
    t_centered = t - t.mean(dim=1, keepdim=True)

    # Compute cross-covariance matrix
    H = torch.matmul(p_centered.transpose(-2, -1), t_centered)

    # SVD for optimal rotation
    try:
        U, S, V = torch.svd(H)

        # Ensure right-handed coordinate system (det(R) = +1)
        d = torch.det(torch.matmul(V, U.transpose(-2, -1)))
        e = torch.eye(3, device=p.device)

        if d < 0:
            e[2, 2] = -1

        rotation = torch.matmul(V, torch.matmul(e, U.transpose(-2, -1)))

        # Apply optimal rotation and measure deviation
        p_aligned = torch.matmul(p_centered, rotation.transpose(-2, -1))
        return torch.sqrt(torch.mean(torch.sum(
            (p_aligned - t_centered) ** 2, dim=-1
        )))

    except RuntimeError:
        # Fallback to standard RMSD if SVD fails to converge
        return torch.sqrt(F.mse_loss(p_centered, t_centered))
