"""Tests for visualization utilities."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from PIL import Image

from depthview.visualization.render import create_3d_plot, create_depth_colormap


class TestDepthColormap:
    def test_returns_pil_image(self, sample_depth):
        result = create_depth_colormap(sample_depth)
        assert isinstance(result, Image.Image)

    def test_output_size_matches_input(self, sample_depth):
        result = create_depth_colormap(sample_depth)
        h, w = sample_depth.shape
        assert result.size == (w, h)

    def test_rgb_mode(self, sample_depth):
        result = create_depth_colormap(sample_depth)
        assert result.mode == "RGB"

    def test_different_colormaps(self, sample_depth):
        img1 = create_depth_colormap(sample_depth, colormap="turbo")
        img2 = create_depth_colormap(sample_depth, colormap="viridis")
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        assert not np.array_equal(arr1, arr2)


class TestCreate3dPlot:
    def test_returns_plotly_figure(self):
        points = np.random.rand(100, 3).astype(np.float32)
        colors = np.random.rand(100, 3).astype(np.float32)
        fig = create_3d_plot(points, colors)
        assert isinstance(fig, go.Figure)

    def test_has_scatter3d_trace(self):
        points = np.random.rand(50, 3).astype(np.float32)
        colors = np.random.rand(50, 3).astype(np.float32)
        fig = create_3d_plot(points, colors)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Scatter3d)

    def test_correct_point_count(self):
        n = 200
        points = np.random.rand(n, 3).astype(np.float32)
        colors = np.random.rand(n, 3).astype(np.float32)
        fig = create_3d_plot(points, colors)
        assert len(fig.data[0].x) == n

    def test_custom_point_size(self):
        points = np.random.rand(10, 3).astype(np.float32)
        colors = np.random.rand(10, 3).astype(np.float32)
        fig = create_3d_plot(points, colors, point_size=5.0)
        assert fig.data[0].marker.size == 5.0
