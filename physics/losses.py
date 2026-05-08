"""
Six-Term Physics-Informed Loss Functions for RNA Structure Validation.

Each term addresses a distinct aspect of RNA conformational energetics:
1. Steric clash repulsion (geometric feasibility)
2. Radius of gyration compaction (global folding)
3. Coulomb electrostatic repulsion (phosphate charge)
4. Torsional strain (backbone rotamer preference)
5. Pi-pi base stacking (aromatic interaction)
6. Hydrogen bonding (Watson-Crick pairing)

All functions are fully differentiable and implemented with PyTorch autograd support.
"""

import torch
import numpy as np


def calculate_L_clash(coords, min_distance=1.5):
    """
    Calculates the soft steric clash penalty for all atom pairs.

    Implements a soft hinge function that penalizes only violations below
    a minimum distance threshold. Adjacent bonded atoms are masked from
    computation to avoid penalizing covalent bonds.

    Args:
        coords (torch.Tensor): Atomic coordinates, shape [N, 3].
        min_distance (float): Exclusion distance threshold in Angstroms.
            Default: 1.5 (conservative van der Waals distance).

    Returns:
        torch.Tensor: Total clash penalty (scalar).

    Note:
        Uses torch.cdist for optimized O(N^2) pairwise distance computation.
        The quadratic penalty ensures smooth differentiability at threshold.
    """
    # Compute full pairwise distance matrix [N, N]
    dist_matrix = torch.cdist(coords, coords)

    N = coords.shape[0]

    # Mask diagonal (self-distances) and adjacent atoms (bonded pairs)
    mask = torch.eye(N, dtype=torch.bool) |            torch.diag(torch.ones(N - 1, dtype=torch.bool), diagonal=1) |            torch.diag(torch.ones(N - 1, dtype=torch.bool), diagonal=-1)

    # Set masked distances to infinity so they are never penalized
    masked_distances = dist_matrix.masked_fill(mask, float('inf'))

    # Soft hinge loss: penalize only distances below threshold
    clash_violations = torch.relu(min_distance - masked_distances)

    # Sum squared violations, divide by 2 for symmetry (A-B = B-A)
    L_clash = torch.sum(clash_violations ** 2) / 2.0

    return L_clash


def calculate_L_Rg(coords):
    """
    Calculates the squared Radius of Gyration (compaction force).

    Measures the spatial extent of the molecule by computing the mean
    squared distance from the center of mass. Minimizing drives the polymer
    from an extended random-coil state toward a compact globular conformation.

    Args:
        coords (torch.Tensor): Atomic coordinates, shape [N, 3].

    Returns:
        torch.Tensor: Squared radius of gyration (scalar).

    Note:
        This term provides a global folding force without imposing specific
        structural templates, mimicking the hydrophobic collapse phase.
    """
    center_of_mass = coords.mean(dim=0)
    rg_sq = torch.mean(torch.sum((coords - center_of_mass) ** 2, dim=1))
    return rg_sq


def calculate_L_coulomb(backbone_coords, epsilon_r=80.0, k_e=332.0):
    """
    Calculates electrostatic repulsion between backbone phosphate groups.

    Each phosphate carries a negative charge, creating substantial repulsion
    that is partially screened by solvent. In vacuum (epsilon_r=1), this term
    dominates and drives phosphate separation. In water (epsilon_r=80),
    it provides a subtler repulsive bias.

    Args:
        backbone_coords (torch.Tensor): Backbone atom coordinates.
        epsilon_r (float): Relative permittivity of solvent.
            Default: 80.0 (approximates aqueous water).
        k_e (float): Coulomb constant in biological units.
            Default: 332.0 (scales output to kcal/mol).

    Returns:
        torch.Tensor: Total electrostatic penalty (scalar).

    Note:
        Assumes every 6th atom represents the negatively charged phosphate (P).
        Uses torch.cdist for efficient GPU batching of pairwise distances.
    """
    # Every 6th atom represents the negatively charged phosphate (P)
    phosphate_coords = backbone_coords[::6]

    num_phosphates = phosphate_coords.shape[0]
    if num_phosphates < 2:
        return torch.tensor(0.0, requires_grad=True)

    # Pairwise distance matrix between all phosphate groups
    distances = torch.cdist(phosphate_coords, phosphate_coords)

    # Prevent division by zero for diagonal (self-distance)
    distances = distances.clone()
    distances.fill_diagonal_(float('inf'))

    # Coulomb's Law: E = k_e * q_i * q_j / (epsilon_r * d)
    # q_i = q_j = -1, so q_i * q_j = +1 (repulsive)
    coulomb_energy = k_e / (epsilon_r * distances)

    # Sum upper triangle to avoid double counting pairs
    total_electrostatic = torch.triu(coulomb_energy, diagonal=1).sum()

    return total_electrostatic


def calculate_L_torsion(predicted_angles, k_torsion=2.0, periodicity=3.0):
    """
    Penalizes backbone dihedral angles at high-energy eclipsed positions.

    Implements a 3-fold periodic potential modeling rotation around
    sp3-hybridized bonds. Creates energy minima at staggered conformations
    and maxima at eclipsed positions.

    Args:
        predicted_angles (torch.Tensor): Predicted torsion angles in radians.
        k_torsion (float): Energy barrier height. Default: 2.0.
        periodicity (float): Number of energy barriers per 2*pi rotation.
            Default: 3.0 (standard for sp3-sp3 bonds).

    Returns:
        torch.Tensor: Total torsional strain energy (scalar).

    Note:
        The phase shift of pi places minima at staggered conformations
        (phi = pi/3, pi, 5*pi/3) which are biochemically preferred.
    """
    phase_shift = torch.tensor(np.pi)

    # 1 + cos(n * phi - pi) creates energy minima at specific biological angles
    torsional_energy = k_torsion * (
        1 + torch.cos(periodicity * predicted_angles - phase_shift)
    )

    total_strain = torch.sum(torsional_energy)
    return total_strain


def calculate_L_stacking(bases_coords, points_per_base=5,
                         ideal_dist=3.4, epsilon=2.0):
    """
    Rewards adjacent nucleobases for stacking at optimal separation.

    Aromatic nucleobases favor stacked arrangements with characteristic
    3.4 Angstrom interplanar spacing. Uses a Lennard-Jones 12-6 functional
    form to create a deep energy well at the optimal distance.

    Args:
        bases_coords (torch.Tensor): Base atom coordinates.
        points_per_base (int): Number of atoms defining each base.
            Default: 5 (simplified planar representation).
        ideal_dist (float): Optimal stacking distance in Angstroms.
            Default: 3.4 (characteristic A-form RNA helix spacing).
        epsilon (float): Well depth in kcal/mol. Default: 2.0.

    Returns:
        torch.Tensor: Total stacking energy (scalar).

    Note:
        The r^-12 term provides strong short-range repulsion preventing
        atomic overlap, while r^-6 provides attractive dispersion forces
        driving long-range ordering into helical arrangements.
    """
    num_bases = bases_coords.shape[0] // points_per_base
    stacking_energy = torch.tensor(0.0, requires_grad=True)

    if num_bases < 2:
        return stacking_energy

    # Compute center of mass for each base
    base_centers = []
    for i in range(num_bases):
        base_points = bases_coords[
            i * points_per_base: (i + 1) * points_per_base
        ]
        base_centers.append(base_points.mean(dim=0))

    base_centers = torch.stack(base_centers)

    # Lennard-Jones 12-6 potential between adjacent bases
    for i in range(num_bases - 1):
        dist = torch.norm(base_centers[i] - base_centers[i + 1])

        # Prevent division by zero if atoms perfectly overlap
        dist = torch.clamp(dist, min=0.1)

        term12 = (ideal_dist / dist) ** 12
        term6 = 2.0 * (ideal_dist / dist) ** 6

        energy = epsilon * (term12 - term6)
        stacking_energy = stacking_energy + energy

    return stacking_energy


def calculate_L_hbond(bases_coords, points_per_base=5,
                      pairs=[(0, 1), (2, 3)], ideal_dist=3.0):
    """
    Rewards complementary base pairs for achieving canonical Watson-Crick distance.

    Hydrogen bonds between complementary bases (A-U via two H-bonds,
    G-C via three) stabilize the secondary structure double helix.
    Uses a harmonic well potential centered at the target distance.

    Args:
        bases_coords (torch.Tensor): Base atom coordinates.
        points_per_base (int): Number of atoms per base representation.
            Default: 5.
        pairs (list): Expected base-pairing partnerships as (i, j) tuples.
            Default: [(0, 1), (2, 3)] for adjacent pairing.
        ideal_dist (float): Target hydrogen bond distance in Angstroms.
            Default: 3.0 (canonical Watson-Crick separation).

    Returns:
        torch.Tensor: Total hydrogen bond penalty (scalar).

    Note:
        This term should activate only during late-stage training when the
        scaffold has sufficiently compacted to bring potentially paired bases
        into proximity. Premature activation can trap structures in non-native
        topologies.
    """
    loss = torch.tensor(0.0, dtype=torch.float32)

    for (i, j) in pairs:
        # Extract atoms for each base
        base_i = bases_coords[
            i * points_per_base: (i + 1) * points_per_base
        ]
        base_j = bases_coords[
            j * points_per_base: (j + 1) * points_per_base
        ]

        # Compute geometric centers
        center_i = base_i.mean(dim=0)
        center_j = base_j.mean(dim=0)

        # Euclidean distance between base centers
        dist = torch.norm(center_i - center_j)

        # Harmonic well: minimum at ideal_dist
        loss = loss + (dist - ideal_dist) ** 2

    return loss
