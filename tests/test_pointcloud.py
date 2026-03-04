"""Tests for depth-to-point-cloud conversion."""

from __future__ import annotations

import numpy as np

from depthview.models.pointcloud import depth_to_pointcloud, subsample


class TestDepthToPointcloud:
    def test_output_shapes(self, sample_depth, sample_rgb):
        points, colors = depth_to_pointcloud(sample_depth, sample_rgb)
        n = sample_depth.shape[0] * sample_depth.shape[1]
        assert points.shape == (n, 3)
        assert colors.shape == (n, 3)

    def test_colors_normalized(self, sample_depth, sample_rgb):
        _, colors = depth_to_pointcloud(sample_depth, sample_rgb)
        assert colors.min() >= 0.0
        assert colors.max() <= 1.0

    def test_fov_affects_spread(self, sample_depth, sample_rgb):
        points_narrow, _ = depth_to_pointcloud(sample_depth, sample_rgb, fov_deg=30)
        points_wide, _ = depth_to_pointcloud(sample_depth, sample_rgb, fov_deg=120)
        # Wider FOV should produce a wider spread in X
        x_range_narrow = points_narrow[:, 0].max() - points_narrow[:, 0].min()
        x_range_wide = points_wide[:, 0].max() - points_wide[:, 0].min()
        assert x_range_wide > x_range_narrow

    def test_z_values_finite(self, sample_depth, sample_rgb):
        points, _ = depth_to_pointcloud(sample_depth, sample_rgb)
        assert np.all(np.isfinite(points))


class TestSubsample:
    def test_reduces_count(self):
        points = np.random.rand(10000, 3)
        colors = np.random.rand(10000, 3)
        pts, cols = subsample(points, colors, max_points=500)
        assert len(pts) == 500
        assert len(cols) == 500

    def test_no_op_when_small(self):
        points = np.random.rand(100, 3)
        colors = np.random.rand(100, 3)
        pts, cols = subsample(points, colors, max_points=500)
        assert len(pts) == 100

    def test_deterministic(self):
        points = np.random.rand(5000, 3)
        colors = np.random.rand(5000, 3)
        pts1, _ = subsample(points, colors, max_points=100, seed=42)
        pts2, _ = subsample(points, colors, max_points=100, seed=42)
        np.testing.assert_array_equal(pts1, pts2)

    def test_different_seeds(self):
        points = np.random.rand(5000, 3)
        colors = np.random.rand(5000, 3)
        pts1, _ = subsample(points, colors, max_points=100, seed=1)
        pts2, _ = subsample(points, colors, max_points=100, seed=2)
        assert not np.array_equal(pts1, pts2)
