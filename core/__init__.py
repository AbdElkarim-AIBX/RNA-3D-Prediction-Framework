"""
Core geometric and neural network modules for RNA 3D structure prediction.

Contains the differentiable NeRF engine, backbone construction,
and Transformer architecture.
"""

from .nerf import calculate_nerf
from .backbone import build_rna_backbone, attach_rigid_bases
from .transformer import RNATransformer

__all__ = [
    "calculate_nerf",
    "build_rna_backbone",
    "attach_rigid_bases",
    "RNATransformer",
]

from .trainer import RNAFoldTrainer
__all__.append('RNAFoldTrainer')
