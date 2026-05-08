"""
Example 4: Loading Real PDB Structures for Training

Shows how to use RealRNADataset to load experimental structures
from PDB files for supervised training.
"""

from rnafold_net.data import RealRNADataset
from torch.utils.data import DataLoader


def main():
    # Path to directory containing .pdb files
    pdb_directory = "./training_data"

    print(f"Loading PDB files from: {pdb_directory}")

    try:
        # Create dataset from PDB files
        dataset = RealRNADataset(pdb_directory=pdb_directory)

        print(f"Found {len(dataset)} structures")

        # Create data loader
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        # Inspect first sample
        seq, coords = dataset[0]
        print(f"\nFirst sample:")
        print(f"  Sequence length: {len(seq)} nucleotides")
        print(f"  Coordinates shape: {coords.shape}")

        # Iterate through dataset
        for i, (batch_seq, batch_coords) in enumerate(loader):
            print(f"Batch {i+1}: seq={batch_seq.shape}, coords={batch_coords.shape}")
            if i >= 2:  # Show first 3 batches
                break

    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo use this example:")
        print("  1. Create a directory named 'training_data'")
        print("  2. Download RNA PDB files (e.g., 1EHZ.pdb)")
        print("  3. Place them in the directory")
        print("  4. Install biopython: pip install biopython")


if __name__ == "__main__":
    main()
