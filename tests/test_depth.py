"""Integration tests for the depth estimation model (requires model download)."""

from __future__ import annotations

import numpy as np
import pytest

from depthview.models.depth import DepthEstimator


@pytest.mark.slow
class TestDepthEstimator:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.estimator = DepthEstimator()

    def test_load_model(self):
        self.estimator.load_model()
        assert self.estimator._model is not None
        assert self.estimator._processor is not None

    def test_estimate_returns_correct_shape(self, sample_image):
        depth = self.estimator.estimate(sample_image)
        assert depth.shape == (sample_image.height, sample_image.width)

    def test_depth_values_normalized(self, sample_image):
        depth = self.estimator.estimate(sample_image)
        assert depth.min() >= 0.0
        assert depth.max() <= 1.0

    def test_depth_dtype(self, sample_image):
        depth = self.estimator.estimate(sample_image)
        assert depth.dtype == np.float32

    def test_lazy_loading(self):
        estimator = DepthEstimator()
        assert estimator._model is None
        # Accessing .model property triggers load
        _ = estimator.model
        assert estimator._model is not None

    def test_repr(self):
        estimator = DepthEstimator()
        r = repr(estimator)
        assert "DepthEstimator" in r
        assert "loaded=False" in r
