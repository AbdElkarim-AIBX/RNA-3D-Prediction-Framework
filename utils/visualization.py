"""
3D Visualization Utilities for RNA Structures.

Provides interactive Plotly-based rendering of predicted RNA geometries
with separate traces for backbone scaffold and nucleobase rings.
"""

import numpy as np


def visualize_rna_tensors(full_molecule_tensor, backbone_len, num_nucs,
                          filename=None, show=True):
    """
    Renders an interactive 3D visualization of the RNA molecule.

    Creates separate traces for the backbone scaffold (cyan/blue) and
    each rigid nucleobase ring (magenta/purple), displayed in a Plotly
    3D scatter plot.

    Args:
        full_molecule_tensor (torch.Tensor): Combined backbone + base
            coordinates, shape [total_atoms, 3].
        backbone_len (int): Number of atoms in the backbone portion.
        num_nucs (int): Number of nucleotides (bases) to render.
        filename (str, optional): If provided, saves HTML to this path.
        show (bool): Whether to display the figure. Default: True.

    Returns:
        plotly.graph_objects.Figure: The generated figure object.

    Note:
        Requires plotly to be installed. Install with:
            pip install plotly

    Example:
        >>> fig = visualize_rna_tensors(full_molecule, backbone.shape[0], 4)
        >>> fig.write_html("rna_structure.html")
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "plotly is required for visualization. "
            "Install with: pip install plotly"
        )

    coords_np = full_molecule_tensor.detach().cpu().numpy()
    backbone_coords = coords_np[:backbone_len]
    bases_coords = coords_np[backbone_len:]

    fig = go.Figure()

    # Backbone scaffold trace
    fig.add_trace(go.Scatter3d(
        x=backbone_coords[:, 0],
        y=backbone_coords[:, 1],
        z=backbone_coords[:, 2],
        mode='lines+markers',
        marker=dict(
            size=6,
            color='cyan',
            line=dict(width=1, color='darkblue')
        ),
        line=dict(color='blue', width=4),
        name='Backbone Scaffold'
    ))

    # Individual base traces
    points_per_base = 5
    for i in range(num_nucs):
        start_idx = i * points_per_base
        end_idx = start_idx + points_per_base

        if end_idx > len(bases_coords):
            break

        base_c = bases_coords[start_idx:end_idx]
        # Close the ring for visualization
        base_c_closed = np.vstack((base_c, base_c[0]))

        fig.add_trace(go.Scatter3d(
            x=base_c_closed[:, 0],
            y=base_c_closed[:, 1],
            z=base_c_closed[:, 2],
            mode='lines+markers',
            marker=dict(size=4, color='magenta'),
            line=dict(color='purple', width=3),
            name=f'Base {i+1}'
        ))

    fig.update_layout(
        title="RNAFold-Net Predicted 3D Structure",
        scene=dict(
            xaxis_title='X (Angstroms)',
            yaxis_title='Y (Angstroms)',
            zaxis_title='Z (Angstroms)',
            aspectmode='data'
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    if filename:
        fig.write_html(filename)

    if show:
        fig.show()

    return fig
