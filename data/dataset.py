"""
Data Loading and Processing for RNA Structure Datasets.

Supports both synthetic data generation for training/testing and
real PDB file parsing for experimental structure validation.

Synthetic Dataset:
    Generates random RNA sequences and corresponding 3D coordinate targets
    for supervised training scenarios.

Real Dataset:
    Parses PDB files to extract true biological sequences and atomic
    coordinates for ground-truth supervised learning.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class RNADataset(Dataset):
    """
    Synthetic RNA dataset for training and testing.

    Generates random nucleotide sequences and simulated 3D coordinate
    targets. In a production environment, targets would come from
    experimental PDB structures or high-quality computational models.

    Args:
        num_samples (int): Number of synthetic structures to generate.
            Default: 500.
        seq_len (int): Length of each RNA sequence in nucleotides.
            Default: 10.
        atoms_per_nuc (int): Number of atoms per nucleotide in target.
            Default: 9 (3 atoms * 3 coordinates).

    Attributes:
        sequences (torch.Tensor): Integer sequences, shape [num_samples, seq_len].
        true_coords (torch.Tensor): Coordinate targets, shape [num_samples, seq_len, atoms_per_nuc].

    Example:
        >>> dataset = RNADataset(num_samples=100, seq_len=10)
        >>> seq, coords = dataset[0]
        >>> print(seq.shape, coords.shape)  # [10] [10, 9]
    """

    def __init__(self, num_samples=500, seq_len=10, atoms_per_nuc=9):
        self.num_samples = num_samples
        self.seq_len = seq_len

        # Random nucleotide sequences (0=A, 1=C, 2=G, 3=U)
        self.sequences = torch.randint(0, 4, (num_samples, seq_len))

        # Simulated ground truth coordinates
        # In production, load from PDB files or MD trajectories
        self.true_coords = torch.randn(num_samples, seq_len, atoms_per_nuc)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Retrieves a single sample.

        Args:
            idx (int): Sample index.

        Returns:
            tuple: (sequence_indices, true_coordinates)
        """
        return self.sequences[idx], self.true_coords[idx]


class RealRNADataset(Dataset):
    """
    Real RNA structure dataset from PDB files.

    Scans a directory for PDB files and extracts sequences and 3D
    coordinates for supervised training on experimental structures.

    Args:
        pdb_directory (str): Path to directory containing .pdb files.
        mapping (dict, optional): Nucleotide name to index mapping.

    Attributes:
        pdb_files (list): Full paths to PDB files.
        parser (Bio.PDB.PDBParser): PDB parser instance.

    Note:
        Requires biopython to be installed. Install with:
            pip install biopython

    Example:
        >>> dataset = RealRNADataset(pdb_directory='./training_data')
        >>> seq, coords = dataset[0]
    """

    def __init__(self, pdb_directory, mapping=None):
        import os

        self.pdb_directory = pdb_directory
        self.mapping = mapping or {'A': 0, 'C': 1, 'G': 2, 'U': 3}

        # Find all PDB files in directory
        self.pdb_files = [
            os.path.join(pdb_directory, f)
            for f in os.listdir(pdb_directory)
            if f.endswith('.pdb')
        ]

        if not self.pdb_files:
            raise ValueError(
                f"No PDB files found in directory: {pdb_directory}"
            )

        # Lazy initialization of parser (imported on first use)
        self._parser = None

    @property
    def parser(self):
        """Lazy initialization of PDB parser."""
        if self._parser is None:
            try:
                from Bio.PDB import PDBParser
                self._parser = PDBParser(QUIET=True)
            except ImportError:
                raise ImportError(
                    "biopython is required for PDB parsing. "
                    "Install with: pip install biopython"
                )
        return self._parser

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        """
        Parses a single PDB file and extracts sequence and coordinates.

        Extracts backbone atoms (P, C4', N9/N1) for each nucleotide.
        Purines (A, G) use N9; pyrimidines (C, U) use N1.

        Args:
            idx (int): File index.

        Returns:
            tuple: (sequence_indices, coordinates)
                sequence_indices: torch.LongTensor of shape [seq_len]
                coordinates: torch.FloatTensor of shape [seq_len, 9]
                    (3 atoms * 3 coordinates per nucleotide)

        Note:
            Residues with missing atoms are skipped to ensure data quality.
        """
        import os

        structure = self.parser.get_structure(
            "RNA", self.pdb_files[idx]
        )

        # Get first model and first chain
        chain = list(structure.get_models())[0].get_list()[0]

        seq = []
        coords = []

        for residue in chain:
            res_name = residue.resname.strip()

            # Standardize names (handle common PDB naming variations)
            if res_name not in self.mapping:
                continue

            try:
                # Extract key backbone/base atoms
                p_coord = residue['P'].get_coord()
                c4_coord = residue["C4'"].get_coord()

                # Purines use N9, pyrimidines use N1
                if res_name in ['A', 'G']:
                    n_coord = residue['N9'].get_coord()
                else:
                    n_coord = residue['N1'].get_coord()

                seq.append(self.mapping[res_name])

                # Flatten 3 atoms into 9-dimensional tensor
                coords.append(np.concatenate([p_coord, c4_coord, n_coord]))

            except KeyError:
                # Skip residues with missing atoms
                continue

        if not seq:
            raise ValueError(
                f"No valid nucleotides found in {self.pdb_files[idx]}"
            )

        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(np.array(coords), dtype=torch.float32)
        )
