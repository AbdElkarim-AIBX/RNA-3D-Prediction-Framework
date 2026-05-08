"""
Physics-informed loss functions and curriculum scheduling.

Implements six-term physics engine:
- Steric clash repulsion
- Radius of gyration compaction
- Coulomb electrostatic repulsion
- Torsional strain
- Pi-pi base stacking
- Hydrogen bonding
"""

from .losses import (
    calculate_L_clash,
    calculate_L_Rg,
    calculate_L_coulomb,
    calculate_L_torsion,
    calculate_L_stacking,
    calculate_L_hbond,
)
from .curriculum import MasterCurriculum

__all__ = [
    "calculate_L_clash",
    "calculate_L_Rg",
    "calculate_L_coulomb",
    "calculate_L_torsion",
    "calculate_L_stacking",
    "calculate_L_hbond",
    "MasterCurriculum",
]
