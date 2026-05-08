"""
Differentiable NeRF (Natural Extension Reference Frame) Geometry Engine.

Converts predicted torsion angles into 3D Cartesian coordinates through
fully differentiable tensor operations, enabling gradient flow from
physical loss terms back to neural network parameters.

Reference:
    Parsons et al. (2005) Journal of Computational Chemistry 26(10)
"""

import torch


def calculate_nerf(a, b, c, r, theta, phi):
    """
    Differentiable NeRF algorithm to calculate Cartesian coordinates of atom D.

    Given three preceding atoms A, B, C, bond length r (C-D),
    bond angle theta (B-C-D), and torsion angle phi (A-B-C-D),
    computes the 3D position of the new atom D.

    All operations use PyTorch tensors with autograd support.

    Args:
        a (torch.Tensor): 3D coordinates of atom A, shape [3].
        b (torch.Tensor): 3D coordinates of atom B, shape [3].
        c (torch.Tensor): 3D coordinates of atom C, shape [3].
        r (torch.Tensor): Bond length between C and D in Angstroms.
        theta (torch.Tensor): Bond angle B-C-D in radians.
        phi (torch.Tensor): Torsion (dihedral) angle A-B-C-D in radians.

    Returns:
        torch.Tensor: 3D coordinates of atom D, shape [3].

    Example:
        >>> atom_A = torch.tensor([0.0, 0.0, 0.0])
        >>> atom_B = torch.tensor([1.5, 0.0, 0.0])
        >>> atom_C = torch.tensor([2.0, 1.2, 0.0])
        >>> r = torch.tensor(1.54)
        >>> theta = torch.tensor(1.91)
        >>> phi = torch.tensor(1.047)  # 60 degrees
        >>> atom_D = calculate_nerf(atom_A, atom_B, atom_C, r, theta, phi)
    """
    # Compute bond vectors between anchor atoms
    bc = c - b
    bc_hat = bc / torch.norm(bc)

    ab = b - a
    ab_hat = ab / torch.norm(ab)

    # Build orthonormal local reference frame at atom C
    n = torch.cross(ab_hat, bc_hat, dim=0)
    n_hat = n / torch.norm(n)

    v_hat = torch.cross(n_hat, bc_hat, dim=0)

    # Compute local coordinates of atom D in spherical coordinates
    d_local = r * torch.stack([
        -torch.cos(theta),
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi)
    ])

    # Transform local coordinates to global frame via rotation matrix
    M = torch.stack([bc_hat, v_hat, n_hat], dim=1)
    d_global = c + torch.matmul(M, d_local)

    return d_global
