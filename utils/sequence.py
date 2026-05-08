"""
Sequence Processing Utilities for RNA Nucleotide Encoding.

Handles conversion between string sequences and numeric tensor representations
suitable for neural network input.
"""

import torch


# Standard RNA nucleotide mapping
NUCLEOTIDE_MAP = {
    'A': 0,  # Adenine
    'C': 1,  # Cytosine
    'G': 2,  # Guanine
    'U': 3,  # Uracil
    'T': 3,  # Thymine (translated to Uracil for RNA)
    'N': 4,  # Unknown / Padding
}

# Reverse mapping for decoding
INDEX_TO_BASE = {v: k for k, v in NUCLEOTIDE_MAP.items()}


def sequence_to_numeric(sequence, mapping=None):
    """
    Converts an RNA sequence string to numeric tensor indices.

    Maps each nucleotide character to an integer index suitable for
    embedding layer input. Handles both uppercase and lowercase input.

    Args:
        sequence (str): RNA sequence using A, C, G, U/T characters.
            Example: "AUCG" or "AUCGAUCG".
        mapping (dict, optional): Custom character-to-index mapping.
            Default uses standard RNA mapping with padding token.

    Returns:
        torch.Tensor: Long tensor of shape [1, seq_len] containing
            integer indices, ready for nn.Embedding input.

    Raises:
        ValueError: If sequence contains characters not in mapping.

    Example:
        >>> numeric_seq = sequence_to_numeric("AUCG")
        >>> print(numeric_seq)  # tensor([[0, 3, 1, 2]])
    """
    if mapping is None:
        mapping = NUCLEOTIDE_MAP

    sequence = sequence.upper().strip()

    # Validate sequence characters
    invalid_chars = set(sequence) - set(mapping.keys())
    if invalid_chars:
        raise ValueError(
            f"Invalid characters in sequence: {invalid_chars}. "
            f"Valid characters: {list(mapping.keys())}"
        )

    indices = [mapping[base] for base in sequence]
    return torch.tensor([indices], dtype=torch.long)


def numeric_to_sequence(indices, mapping=None):
    """
    Converts numeric tensor indices back to RNA sequence string.

    Args:
        indices (torch.Tensor or list): Integer indices.
        mapping (dict, optional): Custom index-to-character mapping.

    Returns:
        str: RNA sequence string.

    Example:
        >>> seq = numeric_to_sequence([0, 3, 1, 2])
        >>> print(seq)  # "AUCG"
    """
    if mapping is None:
        mapping = INDEX_TO_BASE

    if isinstance(indices, torch.Tensor):
        indices = indices.flatten().tolist()

    return ''.join(mapping.get(i, 'N') for i in indices)


def validate_sequence(sequence):
    """
    Validates that a sequence contains only standard RNA nucleotides.

    Args:
        sequence (str): Sequence to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    valid_chars = set(NUCLEOTIDE_MAP.keys()) - {'N'}
    return all(c.upper() in valid_chars for c in sequence)
