"""FastAPI application for depth estimation and point cloud generation."""

from __future__ import annotations

import base64
import io

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image

from depthview.models.depth import DepthEstimator
from depthview.models.pointcloud import depth_to_pointcloud, subsample
from depthview.schemas.depth import (
    DepthRequest,
    DepthResponse,
    HealthResponse,
    PointCloudRequest,
    PointCloudResponse,
)
from depthview.visualization.render import create_depth_colormap


def decode_image(image_base64: str) -> Image.Image:
    """Decode a base64-encoded image string to a PIL Image."""
    try:
        image_bytes = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}") from e


def encode_image(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL Image to a base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_app(estimator: DepthEstimator) -> FastAPI:
    """Create the FastAPI application with a pre-configured depth estimator.

    Args:
        estimator: Loaded DepthEstimator instance.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="depth-view", version="0.1.0")

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            model=estimator.model_name,
            device=estimator.device,
        )

    @app.post("/depth", response_model=DepthResponse)
    def estimate_depth(request: DepthRequest):
        image = decode_image(request.image_base64)
        depth = estimator.estimate(image)
        depth_img = create_depth_colormap(depth)
        return DepthResponse(
            depth_base64=encode_image(depth_img),
            width=image.width,
            height=image.height,
        )

    @app.post("/pointcloud", response_model=PointCloudResponse)
    def generate_pointcloud(request: PointCloudRequest):
        image = decode_image(request.image_base64)
        depth = estimator.estimate(image)

        depth_img = create_depth_colormap(depth, colormap=request.colormap)

        # Resize image to match depth map dimensions if needed
        h, w = depth.shape[:2]
        if image.size != (w, h):
            image = image.resize((w, h), Image.BILINEAR)
        rgb = np.array(image)
        points, colors = depth_to_pointcloud(depth, rgb, fov_deg=request.fov_deg)
        points, colors = subsample(points, colors, max_points=request.max_points)

        return PointCloudResponse(
            points=points.tolist(),
            colors=colors.tolist(),
            num_points=len(points),
            depth_base64=encode_image(depth_img),
        )

    return app


if __name__ == "__main__":
    import hydra
    import uvicorn
    from omegaconf import DictConfig

    from depthview.configs import register_configs

    register_configs()

    @hydra.main(version_base=None, config_path="../configs", config_name="config")
    def main(cfg: DictConfig) -> None:
        estimator = DepthEstimator(
            model_name=cfg.model.model_id,
            device=cfg.model.device,
        )
        estimator.load_model()
        app = create_app(estimator)
        uvicorn.run(app, host=cfg.serving.host, port=cfg.serving.port)

    main()
