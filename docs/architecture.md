# Architecture Documentation

## System Design

RNAFold-Net follows a sequential four-stage pipeline that transforms raw nucleotide sequences into physically-validated 3D atomic coordinates. Each stage is fully differentiable, enabling end-to-end gradient-based optimization.

## Pipeline Stages

### Stage 1: Sequence Tokenization and Embedding

**Input**: RNA sequence string (e.g., "AUCG")

**Process**:
1. Map characters to integer indices (A=0, C=1, G=2, U=3)
2. Project indices to continuous vectors via learned embedding matrix
3. Add positional encodings to preserve sequential ordering

**Output**: Tensor of shape [batch_size, seq_len, d_model]

**Key Design**: Positional encodings are necessary because the Transformer processes all positions in parallel, losing inherent sequential information.

### Stage 2: Transformer Encoder

**Input**: Embedded sequence tensor

**Process**:
1. Multi-head self-attention computes pairwise affinity scores
2. Feed-forward network processes each position independently
3. Layer normalization and residual connections stabilize training
4. Stack N identical layers for hierarchical feature extraction

**Output**: Contextualized representations H of shape [batch_size, seq_len, d_model]

**Key Design**: The attention mechanism identifies potential base-pairing relationships by computing pairwise scores across the sequence, effectively learning an implicit contact map.

**Mathematical Formulation**:

Scaled dot-product attention:
```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

Multi-head attention:
```
MultiHead(X) = Concat(head_1, ..., head_h) * W^O
head_i = Attention(X * W_i^Q, X * W_i^K, X * W_i^V)
```

### Stage 3: MLP Prediction Head

**Input**: Transformer hidden states H

**Process**:
1. Linear projection from d_model to hidden dimension (64)
2. ReLU activation for non-linearity
3. Linear projection to 7 output angles
4. Sigmoid activation squashes to [0, 1]
5. Scale to [0, 2*pi] for physical validity

**Output**: Predicted angles of shape [batch_size, seq_len, 7]

**Angle Organization**:
- alpha, beta, gamma, delta, epsilon, zeta: backbone dihedrals
- chi: glycosidic torsion controlling base orientation

**Key Design**: Using torsion angles as intermediate representation enforces correct local backbone geometry by construction, eliminating the need for expensive bond length and angle constraints during optimization.

### Stage 4: NeRF Geometry Synthesis

**Input**: Seven torsion angles per nucleotide

**Process**:
1. Initialize three anchor atoms in XY plane
2. For each subsequent atom:
   - Compute local orthonormal frame from three preceding atoms
   - Calculate local coordinates from bond length, angle, and torsion
   - Transform to global frame via rotation matrix
3. Graft rigid base templates using chi angle

**Output**: Complete atomic coordinates of shape [total_atoms, 3]

**NeRF Algorithm**:

Given atoms A, B, C and parameters r, theta, phi:

1. Compute bond vectors:
   ```
   bc = c - b
   ab = b - a
   ```

2. Build orthonormal basis at C:
   ```
   n = normalize(cross(ab, bc))
   v = cross(n, bc)
   u = normalize(bc)
   ```

3. Local coordinates of D:
   ```
   d_local = r * [-cos(theta), sin(theta)*cos(phi), sin(theta)*sin(phi)]
   ```

4. Global coordinates:
   ```
   d_global = c + M * d_local
   where M = [u, v, n]
   ```

**Differentiability**: All operations (cross products, normalizations, matrix multiplications) use PyTorch tensor operations, maintaining full autograd support.

### Stage 5: Physics-Informed Loss Computation

**Input**: 3D atomic coordinates

**Process**: Evaluate six independent energy terms

**Output**: Weighted scalar loss value

## Backbone Parameters

Experimental bond lengths and angles from high-resolution X-ray crystallography:

| Bond | Length (A) | Angle (rad) | Description |
|------|-----------|-------------|-------------|
| P-O5' | 1.60 | 1.81 | Phosphate to O5' oxygen |
| O5'-C5' | 1.41 | 2.08 | O5' to C5' carbon |
| C5'-C4' | 1.53 | 1.91 | C5' to C4' carbon |
| C4'-C3' | 1.53 | 1.91 | C4' to C3' carbon |
| C3'-O3' | 1.41 | 2.02 | C3' to O3' oxygen |
| O3'-P | 1.60 | 2.08 | O3' to next phosphate |

## Physics Engine Details

### Steric Clash Repulsion (L_clash)

Soft hinge function penalizing violations below minimum distance:

```
L_clash = sum((min_dist - d_ij)^2 * I(d_ij < min_dist)) / 2
```

Where:
- d_ij = Euclidean distance between atoms i and j
- min_dist = 1.5 Angstroms (van der Waals exclusion)
- I() = indicator function
- Division by 2 corrects for symmetric pairs

Adjacent bonded atoms are masked to avoid penalizing covalent bonds.

### Radius of Gyration Compaction (L_Rg)

Measures spatial extent and drives globular folding:

```
L_Rg = (1/N) * sum(||r_i - r_com||^2)
```

Where r_com is the center of mass. Minimizing pulls all atoms toward the collective center.

### Coulomb Electrostatic Repulsion (L_coulomb)

Pairwise repulsion between negatively charged phosphate groups:

```
L_coulomb = k_e / (epsilon_r * d_ij)
```

Parameters:
- k_e = 332.0 (biological units, kcal/mol)
- epsilon_r = 80.0 (aqueous solvent screening)
- q_i = q_j = -1 (phosphate charges)

In vacuum (epsilon_r=1), dominates and drives phosphate separation. In water, provides subtler repulsive bias.

### Torsional Strain (L_torsion)

3-fold periodic potential for sp3-hybridized bonds:

```
L_torsion = k * sum(1 + cos(n * phi - pi))
```

Parameters:
- k = 2.0 kcal/mol (barrier height)
- n = 3 (periodicity)
- Minima at staggered conformations (phi = pi/3, pi, 5*pi/3)
- Maxima at eclipsed positions

### Pi-Pi Base Stacking (L_stack)

Lennard-Jones 12-6 potential for aromatic base interactions:

```
L_stack = epsilon * sum((sigma/d)^12 - 2*(sigma/d)^6)
```

Parameters:
- epsilon = 2.0 kcal/mol (well depth)
- sigma = 3.4 Angstroms (optimal stacking distance)
- r^-12 prevents atomic overlap at short range
- r^-6 provides attractive dispersion forces

### Hydrogen Bonding (L_hbond)

Harmonic well rewarding Watson-Crick base pairing:

```
L_hbond = sum((||c_i - c_j|| - d_ideal)^2)
```

Parameters:
- d_ideal = 3.0 Angstroms (canonical H-bond distance)
- c_i, c_j = centers of mass of paired bases
- Only evaluates expected pairs from sequence complementarity

## Dynamic Curriculum Scheduling

### Design Philosophy

A naive simultaneous optimization of all six loss terms leads to poor convergence because conflicting forces act at different stages of folding:

1. Early: Steric clashes must resolve before compaction
2. Mid: Compaction brings bases into proximity
3. Late: Hydrogen bonds can only form when bases are close

The curriculum implements temporal phasing inspired by simulated annealing.

### Clash Weight Schedule (Exponential Decay)

```
lambda_clash(t) = lambda_max * exp(-gamma * t)
```

Parameters:
- lambda_max = 100.0 (strong initial repulsion)
- gamma = 5.0 (decay rate)
- t = normalized training progress [0, 1]

Ensures the "untangling force" is strongest at initialization when random angles produce severely clashed conformations.

### Compaction Weight Schedule (Sigmoid Warmup)

```
lambda_rg(t) = lambda_final / (1 + exp(-beta * (t - t0)))
```

Parameters:
- lambda_final = 10.0 (full compaction strength)
- beta = 15.0 (steepness)
- t0 = 0.4 (midpoint, epoch 400 of 1000)

Prevents premature compaction of an extended chain before local steric conflicts are resolved.

### Biological Constraint Scheduling

Hydrogen bonding and base stacking activate only in final training phases:

```
lambda_hbond(t) = 50.0 / (1 + exp(-50 * (t - 0.7)))
lambda_stack(t) = 5.0 / (1 + exp(-40 * (t - 0.5)))
```

Steep sigmoid transitions create sharp on/off behavior at predetermined milestones, effectively switching biological constraints on after global fold establishment.

## Key Design Decisions

### Torsion vs. Direct Coordinate Prediction

Using torsion angles as intermediate representation:
- **Advantages**: Enforces correct local geometry by construction, physically interpretable, reduces effective dimensionality
- **Trade-off**: Requires NeRF reconstruction step, adds computational overhead

### Fully Differentiable Pipeline

Eliminating sampling-based generation:
- **Advantages**: Deterministic gradient descent, no stochastic exploration needed, seamless integration with deep learning
- **Trade-off**: May converge to local minima without simulated annealing temperature schedule

### Dynamic Curriculum vs. Fixed Weights

Phased constraint activation:
- **Advantages**: Prevents premature convergence to invalid minima, mirrors natural folding process, interpretable training dynamics
- **Trade-off**: Introduces hyperparameters requiring tuning for new sequence types

## Memory and Computational Scaling

### Complexity Analysis

| Component | Time | Space | Notes |
|-----------|------|-------|-------|
| Transformer | O(L^2 * d) | O(L * d) | L=seq_len, d=model_dim |
| NeRF Build | O(L * N_atoms) | O(L * N_atoms) | Sequential chain |
| Clash Loss | O(N^2) | O(N^2) | N=total atoms, pairwise |
| Coulomb | O(P^2) | O(P^2) | P=phosphates (~L/6) |
| Rg | O(N) | O(1) | Single pass |
| H-bond | O(B) | O(1) | B=base pairs |

### Scaling Strategies

For sequences > 200 nucleotides:
1. Use gradient checkpointing in Transformer
2. Implement neighbor lists for clash/Coulomb (O(N) instead of O(N^2))
3. Process in overlapping windows for very long RNAs
4. Consider hierarchical architectures (local + global attention)
