# API Reference

## Core Module

### `calculate_nerf(a, b, c, r, theta, phi)`

Differentiable NeRF algorithm for coordinate reconstruction.

**Parameters**:
- `a` (torch.Tensor): Atom A coordinates, shape [3]
- `b` (torch.Tensor): Atom B coordinates, shape [3]
- `c` (torch.Tensor): Atom C coordinates, shape [3]
- `r` (torch.Tensor): Bond length C-D in Angstroms
- `theta` (torch.Tensor): Bond angle B-C-D in radians
- `phi` (torch.Tensor): Torsion angle A-B-C-D in radians

**Returns**:
- torch.Tensor: Atom D coordinates, shape [3]

**Example**:
```python
atom_D = calculate_nerf(atom_A, atom_B, atom_C, 
                        r=1.54, theta=1.91, phi=1.047)
```

---

### `build_rna_backbone(torsions_batch, num_nucs)`

Constructs RNA phosphodiester backbone from predicted torsion angles.

**Parameters**:
- `torsions_batch` (torch.Tensor): Shape [num_nucleotides, 6] containing
  backbone angles (alpha, beta, gamma, delta, epsilon, zeta)
- `num_nucs` (int): Number of nucleotides

**Returns**:
- torch.Tensor: Backbone coordinates, shape [num_atoms, 3]

**Note**: First 3 atoms are anchor points. Total atoms = 3 + 6 * num_nucs.

---

### `attach_rigid_bases(backbone_coords, num_nucs, chi_angles)`

Grafts rigid nucleobase templates onto backbone sugar carbons.

**Parameters**:
- `backbone_coords` (torch.Tensor): Backbone atom coordinates
- `num_nucs` (int): Number of nucleotides
- `chi_angles` (torch.Tensor): Glycosidic torsion per nucleotide, shape [num_nucs]

**Returns**:
- torch.Tensor: Combined backbone + base coordinates

**Note**: Uses simplified pyrimidine template. Extend for mixed A/U/C/G with base-specific templates.

---

### `RNATransformer`

Transformer model for sequence-to-angle prediction.

**Constructor Parameters**:
- `vocab_size` (int): Number of tokens. Default: 5
- `d_model` (int): Embedding dimension. Default: 128
- `n_heads` (int): Attention heads. Default: 8
- `num_layers` (int): Encoder layers. Default: 4
- `max_seq_len` (int): Maximum sequence length. Default: 200
- `output_angles` (int): Output angles per nucleotide. Default: 7

**Methods**:
- `forward(x)`: Input shape [batch, seq_len], output shape [batch, seq_len, output_angles]

---

### `RNAFoldTrainer`

Master training engine integrating all components.

**Constructor Parameters**:
- `model` (RNATransformer): Neural network instance
- `num_nucs` (int): Sequence length in nucleotides
- `total_epochs` (int): Training iterations. Default: 1000
- `learning_rate` (float): Initial LR. Default: 0.05
- `lr_min` (float): Minimum LR for annealing. Default: 0.0001
- `grad_clip` (float): Gradient clipping threshold. Default: 2.0
- `device` (str): Computation device. Default: 'cpu'

**Methods**:
- `fit(numeric_sequence, log_interval=100)`: Run training loop
- `predict(numeric_sequence)`: Generate structure prediction

**Returns**:
- `fit()`: dict with training history
- `predict()`: tuple (backbone_coords, full_molecule_coords)

---

## Physics Module

### `calculate_L_clash(coords, min_distance=1.5)`

Steric clash repulsion penalty.

**Parameters**:
- `coords` (torch.Tensor): Atomic coordinates, shape [N, 3]
- `min_distance` (float): Exclusion threshold in Angstroms. Default: 1.5

**Returns**:
- torch.Tensor: Scalar penalty value

**Behavior**: Returns 0 when all non-bonded pairs exceed min_distance. Quadratic penalty for violations.

---

### `calculate_L_Rg(coords)`

Radius of gyration compaction force.

**Parameters**:
- `coords` (torch.Tensor): Atomic coordinates, shape [N, 3]

**Returns**:
- torch.Tensor: Squared radius of gyration (scalar)

**Behavior**: Decreases as structure becomes more compact. Minimum at single point (all atoms coincident).

---

### `calculate_L_coulomb(backbone_coords, epsilon_r=80.0, k_e=332.0)`

Electrostatic repulsion between phosphate groups.

**Parameters**:
- `backbone_coords` (torch.Tensor): Backbone atom coordinates
- `epsilon_r` (float): Solvent permittivity. Default: 80.0 (water)
- `k_e` (float): Coulomb constant. Default: 332.0

**Returns**:
- torch.Tensor: Scalar penalty value

**Note**: Assumes every 6th atom is phosphate (P). Returns 0 if fewer than 2 phosphates.

---

### `calculate_L_torsion(predicted_angles, k_torsion=2.0, periodicity=3.0)`

Torsional strain energy.

**Parameters**:
- `predicted_angles` (torch.Tensor): Torsion angles in radians
- `k_torsion` (float): Barrier height. Default: 2.0
- `periodicity` (float): Number of barriers. Default: 3.0

**Returns**:
- torch.Tensor: Scalar energy value

**Behavior**: Minimum at staggered conformations (pi/3, pi, 5*pi/3). Maximum at eclipsed (0, 2*pi/3, 4*pi/3).

---

### `calculate_L_stacking(bases_coords, points_per_base=5, ideal_dist=3.4, epsilon=2.0)`

Pi-pi base stacking potential.

**Parameters**:
- `bases_coords` (torch.Tensor): Base atom coordinates
- `points_per_base` (int): Atoms per base. Default: 5
- `ideal_dist` (float): Optimal stacking distance. Default: 3.4
- `epsilon` (float): Well depth. Default: 2.0

**Returns**:
- torch.Tensor: Scalar energy value

**Behavior**: Negative (attractive) near ideal_dist, positive (repulsive) at short range, approaches 0 at long range.

---

### `calculate_L_hbond(bases_coords, points_per_base=5, pairs=[(0,1),(2,3)], ideal_dist=3.0)`

Hydrogen bonding reward.

**Parameters**:
- `bases_coords` (torch.Tensor): Base atom coordinates
- `points_per_base` (int): Atoms per base. Default: 5
- `pairs` (list): Expected base pairs as (i,j) tuples
- `ideal_dist` (float): Target distance. Default: 3.0

**Returns**:
- torch.Tensor: Scalar penalty (0 when all pairs at ideal_dist)

**Note**: Should activate only in late training to avoid trapping in incorrect topologies.

---

### `MasterCurriculum`

Dynamic weight scheduler for phased training.

**Constructor Parameters**:
- `total_epochs` (int): Training duration. Default: 1000

**Methods**:
- `get_weights(epoch)`: Returns tuple (w_clash, w_rg, w_hbond, w_coulomb, w_torsion, w_stack)
- `get_lambda_clash(epoch, lambda_max=100.0, decay_rate=5.0)`: Exponential decay
- `get_lambda_rg(epoch, lambda_final=10.0, steepness=15.0, midpoint=0.4)`: Sigmoid warmup
- `get_lambda_hbond(epoch, lambda_max=50.0, steepness=50.0, activation_point=0.7)`: Late activation
- `get_lambda_stack(epoch, lambda_max=5.0, steepness=40.0, activation_point=0.5)`: Mid activation
- `get_lambda_torsion(epoch, lambda_max=2.0, steepness=40.0, activation_point=0.5)`: Mid activation

---

## Utils Module

### `sequence_to_numeric(sequence, mapping=None)`

Convert RNA sequence string to numeric tensor.

**Parameters**:
- `sequence` (str): RNA sequence using A, C, G, U
- `mapping` (dict, optional): Custom character mapping

**Returns**:
- torch.Tensor: Long tensor of shape [1, seq_len]

**Raises**:
- ValueError: If sequence contains invalid characters

**Example**:
```python
numeric_seq = sequence_to_numeric("AUCG")  # tensor([[0, 3, 1, 2]])
```

---

### `numeric_to_sequence(indices, mapping=None)`

Convert numeric indices back to sequence string.

**Parameters**:
- `indices` (torch.Tensor or list): Integer indices
- `mapping` (dict, optional): Custom index mapping

**Returns**:
- str: RNA sequence

---

### `validate_sequence(sequence)`

Check if sequence contains only standard RNA nucleotides.

**Parameters**:
- `sequence` (str): Sequence to validate

**Returns**:
- bool: True if valid, False otherwise

---

### `calculate_rmsd(pred, target)`

Standard RMSD without alignment.

**Parameters**:
- `pred` (torch.Tensor): Predicted coordinates
- `target` (torch.Tensor): Target coordinates

**Returns**:
- torch.Tensor: RMSD in Angstroms

---

### `calculate_kabsch_rmsd(pred, target)`

RMSD after optimal alignment via Kabsch algorithm.

**Parameters**:
- `pred` (torch.Tensor): Predicted coordinates, shape [batch, atoms, 3]
- `target` (torch.Tensor): Target coordinates, same shape

**Returns**:
- torch.Tensor: Kabsch-aligned RMSD in Angstroms

**Note**: Computes optimal rotation matrix via SVD to minimize RMSD. Standard for scientific structure comparison.

---

### `visualize_rna_tensors(full_molecule_tensor, backbone_len, num_nucs, filename=None, show=True)`

Render interactive 3D visualization.

**Parameters**:
- `full_molecule_tensor` (torch.Tensor): Complete coordinates
- `backbone_len` (int): Number of backbone atoms
- `num_nucs` (int): Number of nucleotides to render
- `filename` (str, optional): Save HTML to this path
- `show` (bool): Display in browser. Default: True

**Returns**:
- plotly.graph_objects.Figure: Figure object

**Requires**: plotly >= 5.0.0

---

## Data Module

### `RNADataset`

Synthetic dataset for training and testing.

**Constructor Parameters**:
- `num_samples` (int): Number of structures. Default: 500
- `seq_len` (int): Sequence length. Default: 10
- `atoms_per_nuc` (int): Atoms per nucleotide in target. Default: 9

**Methods**:
- `__getitem__(idx)`: Returns (sequence, coordinates) tuple

---

### `RealRNADataset`

Experimental structure dataset from PDB files.

**Constructor Parameters**:
- `pdb_directory` (str): Path to directory containing .pdb files
- `mapping` (dict, optional): Nucleotide name mapping

**Methods**:
- `__getitem__(idx)`: Returns (sequence, coordinates) tuple

**Requires**: biopython >= 1.79

**Note**: Extracts P, C4', and N9/N1 atoms per nucleotide. Skips residues with missing atoms.
