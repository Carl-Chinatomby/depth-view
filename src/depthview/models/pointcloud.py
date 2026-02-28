"""Convert depth maps to 3D point clouds via backprojection."""

from __future__ import annotations

import numpy as np


def depth_to_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    fov_deg: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a depth map and RGB image into a colored 3D point cloud.

    Uses synthetic camera intrinsics derived from the assumed field of view.
    The depth map should have values in [0, 1] where 1 is closest to the camera.

    Args:
        depth: Depth map of shape (H, W) with values in [0, 1].
        rgb: RGB image of shape (H, W, 3) with values in [0, 255].
        fov_deg: Assumed horizontal field of view in degrees.

    Returns:
        Tuple of (points, colors) where:
            points: (N, 3) array of 3D coordinates (X, Y, Z).
            colors: (N, 3) array of RGB values normalized to [0, 1].
    """
    h, w = depth.shape[:2]

    # Synthetic camera intrinsics from FOV
    fx = fy = w / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
    cx = w / 2.0
    cy = h / 2.0

    # Pixel grid
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    # Use depth directly as Z (invert so closer = smaller Z for natural 3D viewing)
    z = 1.0 - depth + 0.01  # small offset to avoid z=0

    # Backproject
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, -y, -z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0

    return points, colors


def subsample(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int = 15000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly subsample a point cloud to at most max_points.

    Args:
        points: (N, 3) array of 3D coordinates.
        colors: (N, 3) array of RGB values.
        max_points: Maximum number of points to keep.
        seed: Random seed for reproducibility.

    Returns:
        Subsampled (points, colors) tuple.
    """
    n = len(points)
    if n <= max_points:
        return points, colors

    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=max_points, replace=False)
    return points[indices], colors[indices]
