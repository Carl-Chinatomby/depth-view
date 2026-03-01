"""Visualization utilities for depth maps and 3D point clouds."""

from __future__ import annotations

import matplotlib
import numpy as np
import plotly.graph_objects as go
from PIL import Image

matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402


def create_depth_colormap(
    depth: np.ndarray,
    colormap: str = "turbo",
) -> Image.Image:
    """Render a depth map as a colored image using a matplotlib colormap.

    Args:
        depth: Depth map of shape (H, W) with values in [0, 1].
        colormap: Matplotlib colormap name (e.g., "turbo", "viridis", "plasma").

    Returns:
        PIL Image with the colormap applied.
    """
    cmap = colormaps[colormap]
    colored = cmap(depth)  # (H, W, 4) RGBA float in [0, 1]
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored_rgb)


def create_3d_plot(
    points: np.ndarray,
    colors: np.ndarray,
    point_size: float = 2.0,
) -> go.Figure:
    """Create an interactive 3D scatter plot from a point cloud.

    Args:
        points: (N, 3) array of 3D coordinates (X, Y, Z).
        colors: (N, 3) array of RGB values in [0, 1].
        point_size: Size of each point in the scatter plot.

    Returns:
        Plotly Figure with the 3D point cloud.
    """
    # Convert colors to hex strings for Plotly
    color_hex = [f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})" for r, g, b in colors]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=color_hex,
                    opacity=0.8,
                ),
                hoverinfo="skip",
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgb(20, 20, 20)",
            aspectmode="data",
        ),
        paper_bgcolor="rgb(20, 20, 20)",
        margin=dict(l=0, r=0, t=0, b=0),
        width=600,
        height=500,
    )

    return fig
