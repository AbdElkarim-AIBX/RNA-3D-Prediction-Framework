# RNAFold-Net

Physics-Informed Neural Network Framework for RNA 3D Structure Prediction

Integrates Transformer-based sequence encoding with differentiable NeRF geometry and multi-term physics constraints for end-to-end RNA structure prediction from primary sequence.

## Overview

RNAFold-Net addresses the central challenge of predicting RNA three-dimensional structure from nucleotide sequence through three interconnected innovations:

1. **Transformer Sequence Encoding**: Multi-head self-attention captures long-range nucleotide dependencies, including Watson-Crick and non-canonical base pairing signals, without requiring multiple sequence alignments.

2. **Differentiable NeRF Geometry Engine**: Converts predicted torsion angles into Cartesian coordinates through fully differentiable tensor operations, enabling gradient flow from physical loss terms directly back to neural network parameters.

3. **Six-Term Physics-Informed Loss**: Dynamic curriculum scheduling implements phased training where geometric constraints (clash avoidance, compaction) activate early and biological constraints (hydrogen bonding, base stacking) activate during late-stage refinement.

The system achieves a Kabsch-aligned RMSD of 2.55 Angstroms within 1000 optimization steps, enabling rapid, template-free structural refinement.

## Architecture

```
RNA Sequence (A,U,C,G tokens)
    |
    v
[Transformer Encoder] -- Multi-head Self-Attention
    |
    v
[MLP Predictor] -- 7 Torsion Angles per nucleotide
    |
    v
[NeRF Engine] -- 3D Coordinates
    |
    v
[Physics Loss] -- 6 Energy Terms
```

### Processing Pipeline

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | Sequence Tokenization | Maps A,C,G,U to integer indices with positional encoding |
| 2 | Transformer Encoder | Multi-head self-attention computes contextualized representations |
| 3 | MLP Prediction Head | Projects hidden states to 7 torsion angles (6 backbone + 1 chi) |
| 4 | NeRF Geometry Synthesis | Converts angles to Cartesian coordinates via local frame transformations |
| 5 | Physics Loss Computation | Six energy terms evaluate physical plausibility |

## Installation

### From Source

```bash
git clone https://github.com/yourusername/rnafold-net.git
cd rnafold-net
pip install -e .
```

### With Optional Dependencies

```bash
# For PDB file parsing support
pip install -e ".[pdb]"

# For development and testing
pip install -e ".[dev]"

# For full visualization support
pip install -e ".[viz]"
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- NumPy >= 1.21.0
- plotly >= 5.0.0 (for 3D visualization)
- biopython >= 1.79 (for PDB parsing, optional)

## Quick Start

### Basic Usage

```python
import torch
from rnafold_net import RNATransformer, sequence_to_numeric
from rnafold_net.core import RNAFoldTrainer

# Define your RNA sequence
sequence = "AUCG"
numeric_seq = sequence_to_numeric(sequence)
num_nucs = len(sequence)

# Initialize the model
model = RNATransformer(
    vocab_size=5,
    d_model=128,
    n_heads=8,
    num_layers=4
)

# Create trainer and run optimization
trainer = RNAFoldTrainer(
    model=model,
    num_nucs=num_nucs,
    total_epochs=1000,
    learning_rate=0.05
)

history = trainer.fit(numeric_seq, log_interval=100)

# Generate final prediction
backbone, full_structure = trainer.predict(numeric_seq)

print(f"Backbone atoms: {backbone.shape[0]}")
print(f"Total atoms: {full_structure.shape[0]}")
```

### Visualization

```python
from rnafold_net import visualize_rna_tensors

# Render the predicted structure
fig = visualize_rna_tensors(
    full_structure,
    backbone_len=backbone.shape[0],
    num_nucs=num_nucs,
    filename="predicted_structure.html"
)
```

### Using Real PDB Data for Training

```python
from rnafold_net.data import RealRNADataset
from torch.utils.data import DataLoader

# Load experimental structures
dataset = RealRNADataset(pdb_directory="./training_data")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Use in supervised training loop
for batch_seq, batch_coords in loader:
    # Training code here
    pass
```

## Physics Engine

The framework implements six independent energy terms:

| Term | Description | Activation Phase |
|------|-------------|------------------|
| Steric Clash | Soft hinge penalty for atoms closer than 1.5 Angstroms | Early (exponential decay) |
| Radius of Gyration | Compaction force driving globular folding | Mid (sigmoid warmup) |
| Coulomb Repulsion | Electrostatic repulsion between phosphate groups | Always active |
| Torsional Strain | 3-fold periodic potential for backbone rotamers | Mid (epoch 500) |
| Pi-Pi Stacking | Lennard-Jones 12-6 potential for base stacking | Mid (epoch 500) |
| Hydrogen Bonding | Harmonic well for Watson-Crick base pairing | Late (epoch 700) |

### Curriculum Scheduling

The dynamic curriculum prevents conflicting forces from creating adversarial gradients:

```python
from rnafold_net.physics import MasterCurriculum

scheduler = MasterCurriculum(total_epochs=1000)

# Get weights for any epoch
w_clash, w_rg, w_hbond, w_coulomb, w_torsion, w_stack =     scheduler.get_weights(epoch=500)
```

## API Reference

### Core Components

#### `calculate_nerf(a, b, c, r, theta, phi)`
Differentiable NeRF algorithm for coordinate reconstruction.

#### `build_rna_backbone(torsions_batch, num_nucs)`
Constructs RNA phosphodiester backbone from predicted torsion angles.

#### `attach_rigid_bases(backbone_coords, num_nucs, chi_angles)`
Grafts rigid nucleobase templates onto the backbone.

#### `RNATransformer`
Transformer model for sequence-to-angle prediction.

### Physics Losses

- `calculate_L_clash(coords, min_distance=1.5)` -- Steric clash penalty
- `calculate_L_Rg(coords)` -- Radius of gyration compaction
- `calculate_L_coulomb(backbone_coords, epsilon_r=80.0)` -- Electrostatic repulsion
- `calculate_L_torsion(predicted_angles, k_torsion=2.0)` -- Torsional strain
- `calculate_L_stacking(bases_coords, ideal_dist=3.4)` -- Base stacking potential
- `calculate_L_hbond(bases_coords, pairs, ideal_dist=3.0)` -- Hydrogen bonding reward

### Utilities

- `sequence_to_numeric(sequence)` -- Convert string to tensor indices
- `calculate_rmsd(pred, target)` -- Standard RMSD calculation
- `calculate_kabsch_rmsd(pred, target)` -- Kabsch-aligned RMSD
- `visualize_rna_tensors(...)` -- Interactive 3D visualization

## Training Configuration

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 128 | Transformer embedding dimension |
| `n_heads` | 8 | Number of attention heads |
| `num_layers` | 4 | Transformer encoder layers |
| `learning_rate` | 0.05 | Initial learning rate |
| `total_epochs` | 1000 | Training iterations |
| `grad_clip` | 2.0 | Maximum gradient norm |

### Curriculum Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_clash_max` | 100.0 | Initial clash weight |
| `lambda_rg_final` | 10.0 | Final compaction weight |
| `lambda_hbond_max` | 50.0 | Maximum H-bond weight |
| `activation_midpoint` | 0.4 | Rg sigmoid midpoint (normalized) |
| `hbond_activation` | 0.7 | H-bond activation point (normalized) |

## Results

The framework was evaluated on benchmark RNA sequences:

| Metric | Initial | Final | Improvement |
|--------|---------|-------|-------------|
| Kabsch RMSD | 28.0 Angstroms | 2.55 Angstroms | 90.9% reduction |
| Energy Loss | 2151.19 kcal/mol | 1.72 kcal/mol | 99.9% reduction |

Convergence characteristics:
- Phase I (0-150 epochs): Rapid clash resolution
- Phase II (150-200 epochs): Compaction activation, temporary strain increase
- Phase III (200-1000 epochs): Steady refinement with biological constraints

## Citation

If you use RNAFold-Net in your research, please cite:

```bibtex
@conference{gharib2026rnafold,
  title={RNAFold-Net: A Physics-Informed Neural Network Framework for RNA 3D Structure Prediction},
  author={Gharib, Abdelkarim and Kadi, Imededdine},
  booktitle={The National Conference on Applied Mathematics and Artificial Intelligence (NCMAI)},
  year={2026},
  organization={University of Amar Telidji, Laghouat, Algeria},
  address={Laghouat, Algeria}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

- Author: Abdelkarim Gharib
- Email: a.gharib.inf@lagh-univ.dz
- Supervisor: Kadi Imededdine (kadi.imed.eddine@gmail.com)
- Institution: University of Amar Telidji, Laghouat, Algeria
- Laboratory: Research Unit in Medicinal Plants (URPM)

## Acknowledgments

This work was developed at the Laboratory of Research Unit in Medicinal Plants (URPM), University of Amar Telidji, Laghouat, Algeria, and presented at the National Conference on Applied Mathematics and Artificial Intelligence (NCMAI) 2026.
