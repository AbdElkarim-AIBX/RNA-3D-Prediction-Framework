"""
RNA Backbone and Nucleobase Construction.

Builds the complete RNA phosphodiester backbone from predicted torsion angles
and grafts rigid nucleobase templates onto the sugar carbons.

RNA Backbone Pattern (per nucleotide):
    P -> O5' -> C5' -> C4' -> C3' -> O3' -> next P

Bond parameters derived from high-resolution X-ray crystallography.
"""

import torch
import numpy as np


# Experimental RNA backbone bond parameters from crystallographic data
# Order: [P-O5', O5'-C5', C5'-C4', C4'-C3', C3'-O3', O3'-P]
RNA_BOND_LENGTHS = torch.tensor(
    [1.60, 1.41, 1.53, 1.53, 1.41, 1.60], dtype=torch.float32
)

# Approximate bond angles in radians (~104 to ~116 degrees)
RNA_BOND_ANGLES = torch.tensor(
    [1.81, 2.08, 1.91, 1.91, 2.02, 2.08], dtype=torch.float32
)

# Rigid pyrimidine ring template (Uracil/Cytosine)
# 5-atom planar pentagonal representation
BASE_RADIUS = 1.4
_angles = torch.linspace(0, 2 * np.pi, 6, dtype=torch.float32)[:-1]
RIGID_PYRIMIDINE = torch.stack([
    BASE_RADIUS * torch.cos(_angles),
    BASE_RADIUS * torch.sin(_angles),
    torch.zeros(5, dtype=torch.float32)  # Enforce absolute planarity
], dim=1)


def build_rna_backbone(torsions_batch, num_nucs):
    """
    Builds the RNA phosphodiester backbone from predicted torsion angles.

    Iteratively applies the NeRF algorithm using experimentally-derived
    bond lengths and angles for the 6-atom repeating backbone pattern.

    Args:
        torsions_batch (torch.Tensor): Shape [num_nucleotides, 6] containing
            backbone torsion angles (alpha, beta, gamma, delta, epsilon, zeta).
        num_nucs (int): Number of nucleotides in the sequence.

    Returns:
        torch.Tensor: Atomic coordinates of the complete backbone,
            shape [num_atoms, 3].

    Note:
        The first three atoms are initialized as anchor points in the XY plane
        to establish the initial coordinate frame.
    """
    # Initialize three anchor atoms in the XY plane
    coords = [
        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([1.60, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([
            1.60 - 1.60 * np.cos(1.81),
            1.60 * np.sin(1.81),
            0.0
        ], dtype=torch.float32)
    ]

    flat_torsions = torsions_batch.flatten()

    # Iteratively build the chain using NeRF
    for i, phi in enumerate(flat_torsions):
        idx = i % 6  # Cycle through the 6 bond types
        r = RNA_BOND_LENGTHS[idx]
        theta = RNA_BOND_ANGLES[idx]

        d = calculate_nerf(coords[-3], coords[-2], coords[-1], r, theta, phi)
        coords.append(d)

    return torch.stack(coords)


def attach_rigid_bases(backbone_coords, num_nucs, chi_angles):
    """
    Grafts rigid nucleobase rings onto the backbone via glycosidic chi angle.

    For each nucleotide, identifies the C4' and C3' attachment points,
    places the base origin using NeRF with the chi torsion angle,
    then rotates the planar base template by chi.

    Args:
        backbone_coords (torch.Tensor): Backbone atom coordinates.
        num_nucs (int): Number of nucleotides.
        chi_angles (torch.Tensor): Glycosidic torsion angle per nucleotide.

    Returns:
        torch.Tensor: Combined backbone + base coordinates.

    Note:
        Uses a simplified pyrimidine template. For production use with
        mixed A/U/C/G sequences, extend with purine (adenine/guanine)
        templates and base-specific parameters.
    """
    all_atoms = [backbone_coords]

    for i in range(num_nucs):
        # Locate attachment points on the backbone
        c4_prime_idx = 3 + (i * 6) + 3
        c3_prime_idx = 3 + (i * 6) + 4

        if c3_prime_idx >= backbone_coords.shape[0]:
            break

        c4 = backbone_coords[c4_prime_idx]
        c3 = backbone_coords[c3_prime_idx]

        # Place base origin using NeRF with chi angle
        r_branch = torch.tensor(1.5, dtype=torch.float32)
        theta_branch = torch.tensor(1.9, dtype=torch.float32)
        dummy_anchor = backbone_coords[c3_prime_idx - 2]

        base_origin = calculate_nerf(
            dummy_anchor, c4, c3,
            r_branch, theta_branch, chi_angles[i]
        )

        # Rotate the planar base ring by chi angle
        chi = chi_angles[i]
        cos_chi = torch.cos(chi)
        sin_chi = torch.sin(chi)

        R_chi = torch.tensor([
            [cos_chi, -sin_chi, 0.0],
            [sin_chi,  cos_chi, 0.0],
            [0.0,      0.0,     1.0]
        ], dtype=torch.float32)

        rotated_base = torch.matmul(RIGID_PYRIMIDINE, R_chi.T)
        positioned_base = rotated_base + base_origin
        all_atoms.append(positioned_base)

    return torch.cat(all_atoms, dim=0)


# Import here to avoid circular dependency
from .nerf import calculate_nerf
