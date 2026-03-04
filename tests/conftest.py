"""Shared test fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_image() -> Image.Image:
    """A small RGB test image."""
    return Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))


@pytest.fixture
def sample_depth() -> np.ndarray:
    """A synthetic depth map with values in [0, 1]."""
    h, w = 64, 64
    y, x = np.mgrid[0:h, 0:w]
    depth = ((x / w + y / h) / 2.0).astype(np.float32)
    return depth


@pytest.fixture
def sample_rgb() -> np.ndarray:
    """A small RGB array matching sample_depth dimensions."""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def mock_estimator(sample_depth):
    """A mock DepthEstimator that returns sample_depth without loading a model."""
    estimator = MagicMock()
    estimator.model_name = "mock-model"
    estimator.device = "cpu"
    estimator.estimate.return_value = sample_depth
    return estimator
