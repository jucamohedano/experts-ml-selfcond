#!/usr/bin/env python3
"""
Interactive 3D visualization of brain tessellation using Plotly.
"""

import pathlib
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from selfcond.brain_data import load_brain_data, create_grid_regions

try:
    import plotly.graph_objects as go
except ImportError:
    print("Plotly not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    import plotly.graph_objects as go


def visualize_tessellation_interactive(mat_path: pathlib.Path, 
                                        grid_shape=(5, 5, 4),
                                        output_path: pathlib.Path = None,
                                        sample_fraction: float = 0.5):
    """
    Create interactive 3D visualization of brain tessellation.
    """
    print(f"Loading brain data from {mat_path}...")
    word_activations, coords, word_labels = load_brain_data(mat_path)
    print(f"Loaded {len(word_labels)} words, {coords.shape[0]} voxels")
    
    # Create regions
    print(f"Creating tessellation with grid shape {grid_shape}...")
    regions = create_grid_regions(coords, grid_shape)
    print(f"Created {len(regions)} non-empty regions")
    
    # --- FIX: Handle Tuple Keys Correctly ---
    def get_sort_key(region_tuple):
        # Ensure elements are ints so (1, 2) comes before (1, 10)
        # This handles cases where the tuple might contain strings like ('1', '10')
        return tuple(int(x) for x in region_tuple)

    # Sort the regions naturally
    region_list = sorted(list(regions.keys()), key=get_sort_key)
    
    # Create a lookup map: region_tuple -> sorted_index
    region_to_idx = {name: i for i, name in enumerate(region_list)}
    
    # Assign region IDs based on the SORTED order
    voxel_region_ids = np.zeros(coords.shape[0], dtype=int)
    
    for region_name, voxel_indices in regions.items():
        idx = region_to_idx[region_name]
        voxel_region_ids[voxel_indices] = idx
    # --------------------------------
    
    # Subsample for performance
    n_voxels = coords.shape[0]
    sample_size = int(n_voxels * sample_fraction)
    sample_indices = np.random.choice(n_voxels, size=sample_size, replace=False)
    
    # --- ENHANCEMENT: Format Hover Text ---
    # Create clear labels like "Region: [1, 2, 5]"
    hover_texts = []
    for i in sample_indices:
        r_tuple = region_list[voxel_region_ids[i]]
        # Convert tuple (1, 2, 5) to list string "[1, 2, 5]"
        region_str = str(list(r_tuple))
        # Add helper text to clarify dimensions
        hover_texts.append(
            f"<b>Region: {region_str}</b><br>"
            f"X={r_tuple[0]} (Left-Right)<br>"
            f"Y={r_tuple[1]} (Post-Ant)<br>"
            f"Z={r_tuple[2]} (Inf-Sup)"
        )

    # Create the 3D scatter plot
    fig = go.Figure()
    
    # Add voxels colored by region
    fig.add_trace(go.Scatter3d(
        x=coords[sample_indices, 0],
        y=coords[sample_indices, 1],
        z=coords[sample_indices, 2],
        mode='markers',
        marker=dict(
            size=2,
            # Colors now map to the natural spatial order
            color=voxel_region_ids[sample_indices],
            colorscale='Rainbow', 
            opacity=0.7,
            colorbar=dict(title='Region Index', thickness=15)
        ),
        text=hover_texts,
        hoverinfo='text',
        name='Brain Voxels'
    ))
    
    # Add grid boundaries
    nx, ny, nz = grid_shape
    x_bins = np.linspace(coords[:, 0].min(), coords[:, 0].max(), nx + 1)
    y_bins = np.linspace(coords[:, 1].min(), coords[:, 1].max(), ny + 1)
    z_bins = np.linspace(coords[:, 2].min(), coords[:, 2].max(), nz + 1)
    
    grid_lines_x, grid_lines_y, grid_lines_z = [], [], []
    
    for xi in range(len(x_bins) - 1):
        for yi in range(len(y_bins) - 1):
            for zi in range(len(z_bins) - 1):
                x0, x1 = x_bins[xi], x_bins[xi + 1]
                y0, y1 = y_bins[yi], y_bins[yi + 1]
                z0, z1 = z_bins[zi], z_bins[zi + 1]
                
                # Bottom face
                grid_lines_x.extend([x0, x1, None, x1, x1, None, x1, x0, None, x0, x0, None])
                grid_lines_y.extend([y0, y0, None, y0, y1, None, y1, y1, None, y1, y0, None])
                grid_lines_z.extend([z0, z0, None, z0, z0, None, z0, z0, None, z0, z0, None])
                
                # Top face
                grid_lines_x.extend([x0, x1, None, x1, x1, None, x1, x0, None, x0, x0, None])
                grid_lines_y.extend([y0, y0, None, y0, y1, None, y1, y1, None, y1, y0, None])
                grid_lines_z.extend([z1, z1, None, z1, z1, None, z1, z1, None, z1, z1, None])
                
                # Verticals
                grid_lines_x.extend([x0, x0, None, x1, x1, None, x1, x1, None, x0, x0, None])
                grid_lines_y.extend([y0, y0, None, y0, y0, None, y1, y1, None, y1, y1, None])
                grid_lines_z.extend([z0, z1, None, z0, z1, None, z0, z1, None, z0, z1, None])
    
    fig.add_trace(go.Scatter3d(
        x=grid_lines_x,
        y=grid_lines_y,
        z=grid_lines_z,
        mode='lines',
        line=dict(color='rgba(40, 40, 40, 0.6)', width=3),
        hoverinfo='skip',
        name='Grid'
    ))
    
    # Boundary box
    x_min, x_max = x_bins[0], x_bins[-1]
    y_min, y_max = y_bins[0], y_bins[-1]
    z_min, z_max = z_bins[0], z_bins[-1]
    
    boundary_x = [
        x_min, x_max, None, x_max, x_max, None, x_max, x_min, None, x_min, x_min, None,
        x_min, x_max, None, x_max, x_max, None, x_max, x_min, None, x_min, x_min, None,
        x_min, x_min, None, x_max, x_max, None, x_max, x_max, None, x_min, x_min, None
    ]
    boundary_y = [
        y_min, y_min, None, y_min, y_max, None, y_max, y_max, None, y_max, y_min, None,
        y_min, y_min, None, y_min, y_max, None, y_max, y_max, None, y_max, y_min, None,
        y_min, y_min, None, y_min, y_min, None, y_max, y_max, None, y_max, y_max, None
    ]
    boundary_z = [
        z_min, z_min, None, z_min, z_min, None, z_min, z_min, None, z_min, z_min, None,
        z_max, z_max, None, z_max, z_max, None, z_max, z_max, None, z_max, z_max, None,
        z_min, z_max, None, z_min, z_max, None, z_min, z_max, None, z_min, z_max, None
    ]
    
    fig.add_trace(go.Scatter3d(
        x=boundary_x,
        y=boundary_y,
        z=boundary_z,
        mode='lines',
        line=dict(color='rgba(0, 0, 0, 0.9)', width=5),
        hoverinfo='skip',
        name='Boundary',
        showlegend=False
    ))
    
    region_sizes = [len(v) for v in regions.values()]
    
    fig.update_layout(
        title=dict(
            text=f'<b>Brain Tessellation: {grid_shape[0]}×{grid_shape[1]}×{grid_shape[2]} Grid</b><br>'
                 f'<sup>{len(regions)} regions | {n_voxels:,} voxels ({sample_size:,} shown) | '
                 f'Sorted Naturally [X, Y, Z]</sup>',
            x=0.5
        ),
        scene=dict(
            xaxis_title='X (Left-Right)',
            yaxis_title='Y (Posterior-Anterior)',
            zaxis_title='Z (Inferior-Superior)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=60, b=0),
        width=1200,
        height=800
    )
    
    if output_path:
        fig.write_html(str(output_path), include_plotlyjs='cdn')
        print(f"Saved interactive visualization to {output_path}")
        print(f"Open in browser: file://{output_path.absolute()}")
    
    return fig


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Interactive 3D brain tessellation visualization')
    parser.add_argument('--mat-path', type=str, 
                        default='brain_data/data-science-P1.mat',
                        help='Path to brain data .mat file')
    parser.add_argument('--grid', type=int, nargs=3, default=[2, 10, 5],
                        help='Grid shape (nx, ny, nz)')
    parser.add_argument('--output', type=str, default='brain_tessellation_3d_natural.html',
                        help='Output HTML file path')
    parser.add_argument('--sample', type=float, default=0.5,
                        help='Fraction of voxels to display (0.0-1.0)')
    
    args = parser.parse_args()
    
    project_root = pathlib.Path(__file__).parent.parent
    mat_path = project_root / args.mat_path
    output_path = project_root / args.output
    
    visualize_tessellation_interactive(
        mat_path=mat_path,
        grid_shape=tuple(args.grid),
        output_path=output_path,
        sample_fraction=args.sample
    )


if __name__ == '__main__':
    main()