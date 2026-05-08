"""
RNAFold-Net: A Physics-Informed Neural Network Framework for RNA 3D Structure Prediction

Integrates Transformer-based sequence encoding with differentiable NeRF geometry
and multi-term physics constraints for end-to-end RNA structure prediction.

Author: Abdelkarim Gharib
Supervisor: Kadi Imededdine
Institution: University of Amar Telidji, Laghouat, Algeria
Laboratory: Research Unit in Medicinal Plants (URPM)
"""

__version__ = "1.0.0"
__author__ = "Abdelkarim Gharib"
__email__ = "a.gharib.inf@lagh-univ.dz"

from .core import (
    calculate_nerf,
    build_rna_backbone,
    attach_rigid_bases,
    RNATransformer,
)

from .physics import (
    calculate_L_clash,
    calculate_L_Rg,
    calculate_L_coulomb,
    calculate_L_torsion,
    calculate_L_stacking,
    calculate_L_hbond,
    MasterCurriculum,
)

from .utils import (
    visualize_rna_tensors,
    sequence_to_numeric,
    calculate_rmsd,
    calculate_kabsch_rmsd,
)

__all__ = [
    "calculate_nerf",
    "build_rna_backbone",
    "attach_rigid_bases",
    "RNATransformer",
    "calculate_L_clash",
    "calculate_L_Rg",
    "calculate_L_coulomb",
    "calculate_L_torsion",
    "calculate_L_stacking",
    "calculate_L_hbond",
    "MasterCurriculum",
    "visualize_rna_tensors",
    "sequence_to_numeric",
    "calculate_rmsd",
    "calculate_kabsch_rmsd",
]
