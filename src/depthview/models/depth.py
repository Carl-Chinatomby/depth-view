"""Depth estimation wrapper using Depth Anything V2."""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator:
    """Wraps a Depth Anything V2 model for monocular depth estimation.

    The model produces relative inverse depth maps from single RGB images.
    Depth values are normalized to [0, 1] where 1 is closest to the camera.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    def load_model(self) -> None:
        """Load the depth estimation model and processor."""
        if self._model is not None:
            return
        logger.info("Loading depth model: %s", self.model_name)
        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        logger.info("Depth model loaded on %s", self.device)

    @property
    def processor(self):
        if self._processor is None:
            self.load_model()
        return self._processor

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model

    def estimate(self, image: Image.Image) -> np.ndarray:
        """Estimate depth from a single RGB image.

        Args:
            image: PIL Image in RGB mode.

        Returns:
            Depth map as a float32 numpy array of shape (H, W) with values in [0, 1].
            Higher values are closer to the camera.
        """
        image = image.convert("RGB")
        original_size = image.size  # (W, H)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_depth = outputs.predicted_depth  # (1, H', W')

        # Interpolate back to original image size
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(original_size[1], original_size[0]),  # (H, W)
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_np = depth.cpu().numpy()

        # Normalize to [0, 1]
        d_min = depth_np.min()
        d_max = depth_np.max()
        if d_max - d_min > 0:
            depth_np = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_np = np.zeros_like(depth_np)

        return depth_np.astype(np.float32)

    def __repr__(self) -> str:
        loaded = self._model is not None
        return f"DepthEstimator(model={self.model_name!r}, device={self.device!r}, loaded={loaded})"
