# How to Use RNAFold-Net

Complete guide for installing, running, and extending the RNAFold-Net framework for 3D RNA structure prediction.

## Table of Contents

1. [Installation](#installation)
2. [Basic Prediction Workflow](#basic-prediction-workflow)
3. [Understanding the Output](#understanding-the-output)
4. [Training Modes Explained](#training-modes-explained)
5. [Working with Real Data](#working-with-real-data)
6. [Customizing the Physics Engine](#customizing-the-physics-engine)
7. [Visualization and Analysis](#visualization-and-analysis)
8. [Common Workflows](#common-workflows)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting Guide](#troubleshooting-guide)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/rnafold-net.git
cd rnafold-net
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install the package directly:

```bash
pip install -e .
```

### Step 3: Verify Installation

```python
import rnafold_net
print(rnafold_net.__version__)
```

Expected output: `1.0.0`

## Basic Prediction Workflow

### Complete Example: Predict Structure from Sequence

```python
"""
Minimal working example for RNA 3D structure prediction.
This script demonstrates the complete pipeline from sequence to coordinates.
"""

import torch
from rnafold_net import RNATransformer, sequence_to_numeric
from rnafold_net.core import RNAFoldTrainer
from rnafold_net.utils import visualize_rna_tensors

# ------------------------------------------------------------------
# Step 1: Define your RNA sequence
# ------------------------------------------------------------------
sequence = "AUCG"
print(f"Input sequence: {sequence}")

# ------------------------------------------------------------------
# Step 2: Convert to numeric representation
# ------------------------------------------------------------------
numeric_seq = sequence_to_numeric(sequence)
num_nucs = len(sequence)

# ------------------------------------------------------------------
# Step 3: Initialize the neural network
# ------------------------------------------------------------------
model = RNATransformer(
    vocab_size=5,       # 4 bases + padding token
    d_model=128,        # Embedding dimension
    n_heads=8,          # Attention heads
    num_layers=4,       # Transformer encoder layers
    max_seq_len=200     # Maximum sequence length
)

# ------------------------------------------------------------------
# Step 4: Create training engine
# ------------------------------------------------------------------
trainer = RNAFoldTrainer(
    model=model,
    num_nucs=num_nucs,
    total_epochs=1000,      # Training iterations
    learning_rate=0.05,     # Initial learning rate
    lr_min=0.0001,          # Final learning rate (cosine annealing)
    grad_clip=2.0,          # Gradient clipping threshold
    device='cpu'            # or 'cuda' for GPU
)

# ------------------------------------------------------------------
# Step 5: Run optimization
# ------------------------------------------------------------------
print("Starting optimization...")
history = trainer.fit(numeric_seq, log_interval=100)

# ------------------------------------------------------------------
# Step 6: Extract final structure
# ------------------------------------------------------------------
backbone_coords, full_molecule = trainer.predict(numeric_seq)

print(f"
Prediction complete:")
print(f"  Backbone atoms: {backbone_coords.shape[0]}")
print(f"  Total atoms (with bases): {full_molecule.shape[0]}")
print(f"  Final energy: {history['total_loss'][-1]:.2f} kcal/mol")

# ------------------------------------------------------------------
# Step 7: Visualize
# ------------------------------------------------------------------
fig = visualize_rna_tensors(
    full_molecule,
    backbone_len=backbone_coords.shape[0],
    num_nucs=num_nucs,
    filename="my_rna_structure.html"
)
print("Structure saved to my_rna_structure.html")
```

### Running the Example

Save the above code to `predict.py` and run:

```bash
python predict.py
```

Expected console output:
```
Input sequence: AUCG
Starting optimization...

Epoch    0 | LR: 0.050000
  Total Energy :   XXXXX.XX
  1. Clash     :   XXXXX.XX (raw: XXXX.XX)
  2. Compact   :      XX.XX (raw: XXXX.XX)
  ...

Epoch  100 | LR: 0.04XXXX
  ...

Epoch  500 | LR: 0.02XXXX
   >> Stage 3: Torsion and Stacking Activated
  ...

Epoch  700 | LR: 0.00XXXX
   >> Stage 4: Biological Base-Pairing Activated
  ...

Epoch  999 | LR: 0.000100
  Total Energy :       1.72
  ...

Training complete.
Prediction complete:
  Backbone atoms: 27
  Total atoms (with bases): 47
  Final energy: 1.72 kcal/mol
Structure saved to my_rna_structure.html
```

## Understanding the Output

### Training History

The `history` dictionary returned by `fit()` contains:

```python
history = {
    'total_loss': [...],      # Weighted sum of all terms per epoch
    'clash': [...],           # Weighted clash contribution
    'rg': [...],              # Weighted compaction contribution
    'coulomb': [...],         # Weighted electrostatic contribution
    'torsion': [...],          # Weighted torsion contribution
    'stacking': [...],        # Weighted stacking contribution
    'hbond': [...],           # Weighted H-bond contribution
    'learning_rate': [...],   # LR at each epoch
}
```

### Coordinate Tensors

The `predict()` method returns two tensors:

1. **backbone_coords**: Shape `[num_backbone_atoms, 3]`
   - Contains P, O5', C5', C4', C3', O3' atoms for each nucleotide
   - Plus 3 initial anchor atoms
   - Total: 3 + 6 * num_nucs atoms

2. **full_molecule**: Shape `[total_atoms, 3]`
   - Backbone + rigid base templates
   - Total: backbone_atoms + 5 * num_nucs atoms

### Visualization Output

The HTML file contains an interactive 3D plot with:
- **Blue lines**: Phosphodiester backbone path
- **Cyan markers**: Backbone atoms
- **Purple rings**: Nucleobase positions
- **Magenta markers**: Base atoms

Open the HTML file in any web browser to rotate, zoom, and pan.

## Training Modes Explained

### Mode 1: Physics-Only Folding (No Ground Truth)

Use when you have a sequence but no known structure. The physics engine guides folding entirely.

```python
from rnafold_net.core import RNAFoldTrainer

trainer = RNAFoldTrainer(model, num_nucs=4)
history = trainer.fit(numeric_sequence)
```

**When to use**: Novel sequences, synthetic RNAs, sequences without experimental structures.

**Advantages**: No data required, works for any sequence length.

**Limitations**: Accuracy depends on physics parameter quality; may not capture sequence-specific features.

### Mode 2: Supervised Training with Synthetic Data

Generate artificial training data for method development.

```python
from rnafold_net.data import RNADataset
from torch.utils.data import DataLoader

# Create synthetic dataset
dataset = RNADataset(
    num_samples=500,    # Number of structures
    seq_len=10,         # Sequence length
    atoms_per_nuc=9     # 3 atoms * 3 coordinates
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Standard PyTorch training loop
for epoch in range(100):
    for batch_seq, batch_truth in loader:
        optimizer.zero_grad()

        predictions = model(batch_seq)
        loss = torch.nn.functional.mse_loss(predictions, batch_truth)

        loss.backward()
        optimizer.step()
```

**When to use**: Developing new architectures, testing hyperparameters, benchmarking.

**Advantages**: Fast iteration, controlled data distribution.

**Limitations**: Synthetic data may not capture real RNA physics.

### Mode 3: Supervised Training with PDB Data

Train on experimentally determined structures.

```python
from rnafold_net.data import RealRNADataset

# Load experimental structures
dataset = RealRNADataset(pdb_directory="./pdb_files")
loader = DataLoader(dataset, batch_size=4)

# Train with real data
for epoch in range(200):
    for seq, coords in loader:
        predictions = model(seq)

        # Physics-supervised hybrid loss
        physics_loss = calculate_L_clash(predictions.view(-1, 3))
        data_loss = torch.nn.functional.mse_loss(predictions, coords)

        total_loss = 0.7 * data_loss + 0.3 * physics_loss
        total_loss.backward()
        optimizer.step()
```

**When to use**: Production models, benchmark comparisons, fine-tuning.

**Advantages**: Captures real structural preferences, validated by experiment.

**Limitations**: Requires curated PDB dataset, limited to solved structures.

### Mode 4: Hybrid Physics + Supervision

Combine both approaches for robust training.

```python
# In training loop
predictions = model(sequence)
coords_3d = predictions.view(-1, 3)

# Supervised component
data_loss = F.mse_loss(predictions, ground_truth)

# Physics components
clash = calculate_L_clash(coords_3d)
rg = calculate_L_Rg(coords_3d)

# Weighted combination (adjust weights as needed)
total_loss = 0.6 * data_loss + 0.2 * clash + 0.2 * rg
```

**When to use**: Limited experimental data with physics regularization.

**Advantages**: Leverages both data and physical principles.

**Limitations**: Requires tuning of loss weighting.

## Working with Real Data

### Preparing PDB Files

1. Download RNA structures from RCSB PDB (https://www.rcsb.org/)
2. Filter for RNA-only or RNA-protein complexes
3. Place `.pdb` files in a directory

Example directory structure:
```
pdb_files/
  1ehz.pdb
  1fir.pdb
  2tra.pdb
  ...
```

### Loading and Training

```python
from rnafold_net.data import RealRNADataset
from torch.utils.data import DataLoader

# Create dataset
dataset = RealRNADataset(pdb_directory="pdb_files")
print(f"Loaded {len(dataset)} structures")

# Create data loader
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Training loop
for epoch in range(100):
    for batch_seq, batch_coords in loader:
        # batch_seq: [batch_size, seq_len] integer indices
        # batch_coords: [batch_size, seq_len, 9] coordinates

        predictions = model(batch_seq)
        loss = F.mse_loss(predictions, batch_coords)

        loss.backward()
        optimizer.step()
```

### Data Quality Notes

- Missing atoms: Residues with missing backbone atoms are skipped
- Modified nucleotides: Non-standard bases are filtered out
- Multiple chains: Only the first chain is processed
- Resolution: Higher resolution structures (< 3.0A) preferred

## Customizing the Physics Engine

### Modifying Curriculum Schedules

Create a custom curriculum for specific sequence types:

```python
from rnafold_net.physics import MasterCurriculum
import numpy as np

class LongRNACurriculum(MasterCurriculum):
    """
    Extended curriculum for long RNA sequences (> 100 nt).
    Slower clash decay and delayed compaction to handle
    complex folding landscapes.
    """

    def get_lambda_clash(self, epoch, lambda_max=150.0, decay_rate=3.0):
        # Slower decay for extended untangling
        t = epoch / self.total
        return lambda_max * np.exp(-decay_rate * t)

    def get_lambda_rg(self, epoch, lambda_final=8.0, 
                      steepness=10.0, midpoint=0.5):
        # Delayed compaction for long sequences
        t = epoch / self.total
        return lambda_final / (1.0 + np.exp(-steepness * (t - midpoint)))

    def get_lambda_hbond(self, epoch, lambda_max=30.0,
                         steepness=30.0, activation_point=0.75):
        # Later activation for complex tertiary structures
        t = epoch / self.total
        return lambda_max / (1.0 + np.exp(-steepness * (t - activation_point)))

# Use custom curriculum
trainer = RNAFoldTrainer(model, num_nucs=150)
trainer.curriculum = LongRNACurriculum(total_epochs=2000)
trainer.fit(sequence)
```

### Adjusting Physics Parameters

Modify loss function parameters for different solvent conditions:

```python
# High salt conditions (reduced electrostatic repulsion)
loss_coulomb = calculate_L_coulomb(
    backbone_coords, 
    epsilon_r=120.0,  # Higher screening
    k_e=332.0
)

# Vacuum folding (strong repulsion)
loss_coulomb = calculate_L_coulomb(
    backbone_coords,
    epsilon_r=1.0,    # No screening
    k_e=332.0
)

# Tighter steric constraints
loss_clash = calculate_L_clash(
    coords,
    min_distance=2.0  # More conservative than default 1.5
)
```

### Adding Custom Loss Terms

Extend the physics engine with additional constraints:

```python
def calculate_L_custom(coords, target_shape):
    """
    Custom loss term for specific structural features.
    Example: penalize deviation from expected loop geometry.
    """
    # Your custom physics here
    return custom_loss

# Integrate into training loop
loss_custom = calculate_L_custom(full_molecule, expected_loop)

total_loss = (
    w_clash * loss_clash +
    w_rg * loss_rg +
    w_custom * loss_custom  # Your term
)
```

## Visualization and Analysis

### Interactive 3D Visualization

```python
from rnafold_net.utils import visualize_rna_tensors

# Basic visualization
fig = visualize_rna_tensors(
    full_molecule,
    backbone_len=backbone.shape[0],
    num_nucs=4,
    filename="structure.html",
    show=True
)

# Customize appearance
fig.update_layout(
    title="My RNA Structure",
    scene=dict(
        bgcolor='black',
        xaxis=dict(color='white'),
        yaxis=dict(color='white'),
        zaxis=dict(color='white')
    )
)
```

### Plotting Training Curves

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Total loss
axes[0, 0].plot(history['total_loss'])
axes[0, 0].set_title('Total Loss')
axes[0, 0].set_yscale('log')

# Individual terms
terms = ['clash', 'rg', 'coulomb', 'torsion', 'stacking', 'hbond']
for i, term in enumerate(terms):
    row, col = (i + 1) // 3, (i + 1) % 3
    axes[row, col].plot(history[term])
    axes[row, col].set_title(f'{term.capitalize()} Loss')

plt.tight_layout()
plt.savefig('training_curves.png')
```

### Structural Metrics

```python
from rnafold_net.utils import calculate_kabsch_rmsd

# Compare prediction to experimental structure
rmsd = calculate_kabsch_rmsd(predicted_coords, experimental_coords)
print(f"Kabsch RMSD: {rmsd.item():.2f} Angstroms")

# Quality assessment
if rmsd < 2.0:
    print("High accuracy (atomic resolution)")
elif rmsd < 3.0:
    print("Good accuracy (near-atomic)")
elif rmsd < 5.0:
    print("Moderate accuracy")
else:
    print("Low accuracy, needs refinement")
```

## Common Workflows

### Workflow 1: Batch Prediction for Multiple Sequences

```python
sequences = ["AUCG", "GCGC", "UUUAAA", "AUCGAUCG"]

for seq in sequences:
    numeric = sequence_to_numeric(seq)
    model = RNATransformer()
    trainer = RNAFoldTrainer(model, num_nucs=len(seq), total_epochs=500)

    history = trainer.fit(numeric, log_interval=50)
    backbone, full = trainer.predict(numeric)

    # Save coordinates
    torch.save(full, f"{seq}_coords.pt")

    print(f"{seq}: Final loss = {history['total_loss'][-1]:.2f}")
```

### Workflow 2: Hyperparameter Search

```python
import itertools

# Define search space
param_grid = {
    'd_model': [64, 128, 256],
    'n_heads': [4, 8],
    'learning_rate': [0.01, 0.05, 0.1]
}

best_loss = float('inf')
best_params = None

for d_model, n_heads, lr in itertools.product(
    param_grid['d_model'],
    param_grid['n_heads'],
    param_grid['learning_rate']
):
    model = RNATransformer(d_model=d_model, n_heads=n_heads)
    trainer = RNAFoldTrainer(model, num_nucs=4, learning_rate=lr)

    history = trainer.fit(numeric_seq)
    final_loss = history['total_loss'][-1]

    if final_loss < best_loss:
        best_loss = final_loss
        best_params = {'d_model': d_model, 'n_heads': n_heads, 'lr': lr}

print(f"Best parameters: {best_params}")
print(f"Best loss: {best_loss:.2f}")
```

### Workflow 3: Ensemble Prediction

```python
# Train multiple models with different initializations
predictions = []

for i in range(5):
    model = RNATransformer()
    trainer = RNAFoldTrainer(model, num_nucs=4)
    history = trainer.fit(numeric_seq)

    _, full = trainer.predict(numeric_seq)
    predictions.append(full)

# Average ensemble
mean_structure = torch.stack(predictions).mean(dim=0)
std_structure = torch.stack(predictions).std(dim=0)

print(f"Ensemble mean coordinates: {mean_structure.shape}")
print(f"Coordinate uncertainty (std): {std_structure.mean():.3f}")
```

## Performance Optimization

### GPU Acceleration

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

trainer = RNAFoldTrainer(
    model=model,
    num_nucs=4,
    device=device
)
```

### Memory Management for Long Sequences

```python
# For sequences > 100 nucleotides
import torch

# Enable gradient checkpointing in Transformer
model.transformer.gradient_checkpointing = True

# Process in chunks if needed
chunk_size = 50
for i in range(0, len(long_sequence), chunk_size):
    chunk = long_sequence[i:i+chunk_size]
    # Process chunk
```

### Reducing Computation Time

```python
# Reduce epochs for faster iteration
trainer = RNAFoldTrainer(model, num_nucs=4, total_epochs=500)

# Reduce model size for prototyping
model = RNATransformer(d_model=64, n_heads=4, num_layers=2)

# Use mixed precision training (requires GPU)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    predictions = model(sequence)
    loss = compute_loss(predictions)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Troubleshooting Guide

### Problem: Training loss does not decrease

**Symptoms**: Total energy stays high across all epochs.

**Diagnosis**:
```python
# Check individual terms
print(f"Clash: {history['clash'][-1]}")
print(f"Rg: {history['rg'][-1]}")
print(f"H-bond: {history['hbond'][-1]}")
```

**Solutions**:
1. Increase learning rate: `lr=0.1`
2. Extend clash phase: `decay_rate=2.0`
3. Check for NaN values: `torch.isnan(coords).any()`
4. Verify sequence encoding: `print(numeric_seq)`

### Problem: Structure is over-compacted

**Symptoms**: All atoms collapse to a single point, high clash loss.

**Solutions**:
1. Reduce Rg weight: `lambda_final=5.0`
2. Delay Rg activation: `midpoint=0.5`
3. Increase clash weight: `lambda_max=150.0`
4. Check clash threshold: `min_distance=2.0`

### Problem: Hydrogen bonds never form

**Symptoms**: H-bond loss remains high, bases far apart.

**Solutions**:
1. Verify Rg is activating and driving compaction
2. Check stacking is bringing bases into proximity
3. Extend training duration: `total_epochs=1500`
4. Verify sequence has complementary pairs (A-U, G-C)
5. Reduce H-bond activation threshold: `activation_point=0.6`

### Problem: Gradients explode (NaN loss)

**Symptoms**: Loss becomes NaN, training crashes.

**Solutions**:
1. Reduce learning rate: `lr=0.01`
2. Tighten gradient clip: `grad_clip=1.0`
3. Add numerical stability checks:
   ```python
   coords = torch.nan_to_num(coords, nan=0.0, posinf=10.0, neginf=-10.0)
   ```
4. Check for division by zero in custom loss terms

### Problem: Visualization shows broken backbone

**Symptoms**: Disconnected segments, impossible geometry.

**Diagnosis**:
```python
# Check for large coordinate jumps
diffs = torch.diff(backbone_coords, dim=0)
jumps = torch.norm(diffs, dim=1)
print(f"Max jump: {jumps.max():.2f} (should be ~1.4-1.6)")
```

**Solutions**:
1. Increase clash weight to prevent atomic overlap
2. Add bond length constraints to loss function
3. Check NeRF implementation for numerical issues
4. Reduce learning rate for smoother optimization

### Problem: Out of memory on GPU

**Symptoms**: CUDA out of memory error.

**Solutions**:
1. Reduce batch size to 1
2. Use CPU instead: `device='cpu'`
3. Reduce model size: `d_model=64`
4. Enable gradient checkpointing
5. Process shorter sequences in chunks

### Problem: Slow training on CPU

**Symptoms**: Each epoch takes minutes.

**Solutions**:
1. Use GPU if available
2. Reduce total epochs: `total_epochs=500`
3. Reduce model size for prototyping
4. Use smaller sequences for testing
5. Consider Google Colab for free GPU access

## Advanced Topics

### Extending to Proteins

The NeRF engine and physics framework can be adapted for protein structure prediction by:
1. Changing bond parameters to protein values (C-N: 1.33A, N-CA: 1.46A, CA-C: 1.53A)
2. Adding Ramachandran constraints to torsion loss
3. Implementing side chain rotamer libraries
4. Adding hydrophobic burial terms

### Multi-State Ensemble Prediction

For RNAs with multiple conformations:
```python
# Train multiple models with different random seeds
ensemble = []
for seed in range(10):
    torch.manual_seed(seed)
    model = RNATransformer()
    trainer = RNAFoldTrainer(model, num_nucs=4)
    history = trainer.fit(sequence)
    _, coords = trainer.predict(sequence)
    ensemble.append(coords)

# Cluster by structural similarity
from sklearn.cluster import KMeans
# ... clustering code ...
```

### Integration with Molecular Dynamics

Use predicted structures as starting points for MD:
```python
# Save as PDB format
def save_as_pdb(coords, filename):
    with open(filename, 'w') as f:
        for i, (x, y, z) in enumerate(coords):
            f.write(f"ATOM  {i+1:5d}  P   RNA A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}
")

save_as_pdb(full_molecule, "starting_structure.pdb")
```

Then use with GROMACS, AMBER, or OpenMM for refinement.

## Getting Help

- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory
- **Issues**: Report on GitHub Issues page
- **Email**: a.gharib.inf@lagh-univ.dz

## Citation

If you use RNAFold-Net in your research, please cite:

```bibtex
@conference{gharib2026rnafold,
  title={RNAFold-Net: A Physics-Informed Neural Network Framework for RNA 3D Structure Prediction},
  author={Gharib, Abdelkarim and Kadi, Imededdine},
  booktitle={NCMAI 2026},
  year={2026},
  organization={University of Amar Telidji, Laghouat, Algeria}
}
```
