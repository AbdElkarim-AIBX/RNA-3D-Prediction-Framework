# Usage Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Training Modes](#training-modes)
5. [Configuration](#configuration)
6. [Visualization](#visualization)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.12 or higher
- CUDA (optional, for GPU acceleration)

### Install from Source

```bash
git clone https://github.com/yourusername/rnafold-net.git
cd rnafold-net
pip install -e .
```

### Verify Installation

```python
import rnafold_net
print(rnafold_net.__version__)
```

## Quick Start

### Predict Structure from Sequence

```python
from rnafold_net import RNATransformer, sequence_to_numeric
from rnafold_net.core import RNAFoldTrainer

# Your RNA sequence
sequence = "AUCG"
numeric_seq = sequence_to_numeric(sequence)

# Build and train
model = RNATransformer()
trainer = RNAFoldTrainer(model, num_nucs=len(sequence))
history = trainer.fit(numeric_seq)

# Get coordinates
backbone, full = trainer.predict(numeric_seq)
```

## Core Concepts

### Sequence Encoding

RNA sequences use single-letter codes:
- A = Adenine (0)
- C = Cytosine (1)
- G = Guanine (2)
- U = Uracil (3)

The `sequence_to_numeric()` function handles conversion automatically.

### Torsion Angles

Each nucleotide outputs 7 angles:
- 6 backbone dihedrals (alpha, beta, gamma, delta, epsilon, zeta)
- 1 glycosidic torsion (chi)

These angles define the complete 3D conformation through the NeRF engine.

### Physics Terms

Six energy terms guide optimization:

| Term | Purpose | When Active |
|------|---------|-------------|
| Clash | Prevent atomic overlap | Early, strong then decaying |
| Rg | Drive compaction | Mid, sigmoid ramp |
| Coulomb | Phosphate repulsion | Always |
| Torsion | Favor staggered conformations | Mid |
| Stacking | Base-base attraction | Mid |
| H-bond | Watson-Crick pairing | Late |

## Training Modes

### Mode 1: Physics-Only (Unsupervised)

No ground truth required. The physics engine alone guides folding.

```python
trainer = RNAFoldTrainer(model, num_nucs=4)
trainer.fit(numeric_sequence)
```

Best for: Novel sequences without known structures.

### Mode 2: Supervised with Synthetic Data

Train on generated data with coordinate targets.

```python
from rnafold_net.data import RNADataset
from torch.utils.data import DataLoader

dataset = RNADataset(num_samples=500, seq_len=10)
loader = DataLoader(dataset, batch_size=16)

# Standard PyTorch training loop
for epoch in range(100):
    for seq, coords in loader:
        # Your training code
        pass
```

Best for: Method development and validation.

### Mode 3: Supervised with PDB Data

Train on experimental structures.

```python
from rnafold_net.data import RealRNADataset

dataset = RealRNADataset(pdb_directory="./pdb_files")
```

Best for: Production models with experimental validation.

### Mode 4: Hybrid (Physics + Supervision)

Combine physics constraints with supervised loss.

```python
# 80% supervised MSE + 20% physics clash penalty
master_loss = 0.8 * mse_loss + 0.2 * clash_penalty
```

Best for: Fine-tuning with limited experimental data.

## Configuration

### Model Architecture

```python
model = RNATransformer(
    vocab_size=5,      # 4 bases + padding
    d_model=128,       # Embedding dimension (64, 128, 256, 512)
    n_heads=8,         # Attention heads (must divide d_model)
    num_layers=4,      # Encoder depth (2-8 typical)
    max_seq_len=200    # Maximum sequence length
)
```

Guidelines:
- d_model: 128 for small sequences, 256+ for long RNAs
- n_heads: 8 standard, 16 for large models
- num_layers: 4 for basic, 6-8 for complex structures

### Training Parameters

```python
trainer = RNAFoldTrainer(
    model=model,
    num_nucs=76,           # Sequence length
    total_epochs=1000,      # Training iterations
    learning_rate=0.05,   # Initial LR (0.01-0.1 typical)
    lr_min=0.0001,        # Final LR for cosine annealing
    grad_clip=2.0         # Gradient clipping threshold
)
```

### Curriculum Customization

```python
from rnafold_net.physics import MasterCurriculum

class MyCurriculum(MasterCurriculum):
    def get_lambda_clash(self, epoch, lambda_max=200.0, decay_rate=2.0):
        # Slower decay for complex structures
        t = epoch / self.total
        return lambda_max * np.exp(-decay_rate * t)
```

## Visualization

### Interactive 3D Plot

```python
from rnafold_net import visualize_rna_tensors

fig = visualize_rna_tensors(
    full_molecule,
    backbone_len=backbone.shape[0],
    num_nucs=4,
    filename="structure.html",  # Save to file
    show=True                   # Display in browser
)
```

Output:
- Backbone: cyan lines with blue markers
- Bases: magenta rings with purple edges
- Interactive rotation, zoom, pan

### Training Curves

```python
import matplotlib.pyplot as plt

plt.plot(history['total_loss'])
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.show()
```

## Troubleshooting

### Issue: High clash loss persists

**Cause**: Structure remains entangled.

**Solutions**:
- Increase clash weight: `lambda_max=150.0`
- Extend early phase: more epochs before compaction
- Check initial angles: ensure randomization is uniform

### Issue: Structure over-compacts

**Cause**: Rg weight too high too early.

**Solutions**:
- Delay Rg activation: `midpoint=0.5`
- Reduce final weight: `lambda_final=5.0`
- Increase clash decay rate for faster resolution

### Issue: Gradients explode

**Cause**: Physics terms create extreme values.

**Solutions**:
- Reduce learning rate: `lr=0.01`
- Tighten gradient clip: `grad_clip=1.0`
- Check for NaN in coordinates (add `torch.nan_to_num`)

### Issue: H-bonds never form

**Cause**: Bases never come into proximity.

**Solutions**:
- Verify Rg is driving compaction
- Check that stacking is activating
- Ensure sequence has complementary pairs
- Extend training: `total_epochs=1500`

### Issue: CUDA out of memory

**Cause**: Sequence too long for GPU memory.

**Solutions**:
- Reduce batch size
- Use gradient checkpointing
- Process in chunks
- Fall back to CPU

## Performance Tips

1. **Start small**: Test on 4-10 nt sequences before scaling
2. **Monitor curves**: Log every 100 epochs to catch issues early
3. **Save checkpoints**: Save model state every 200 epochs
4. **Validate visually**: Always inspect 3D structure, not just loss values
5. **Compare metrics**: Track both total loss and individual physics terms
