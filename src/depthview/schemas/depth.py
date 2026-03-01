"""Pydantic request/response schemas for the depth estimation API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    device: str


class DepthRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded JPEG or PNG image")
    fov_deg: float = Field(default=60.0, ge=10.0, le=170.0, description="Assumed horizontal FOV")


class DepthResponse(BaseModel):
    depth_base64: str = Field(..., description="Base64-encoded depth colormap PNG")
    width: int
    height: int


class PointCloudRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded JPEG or PNG image")
    fov_deg: float = Field(default=60.0, ge=10.0, le=170.0)
    max_points: int = Field(default=15000, ge=100, le=100000)
    colormap: str = Field(default="turbo")


class PointCloudResponse(BaseModel):
    points: list[list[float]] = Field(..., description="List of [x, y, z] coordinates")
    colors: list[list[float]] = Field(..., description="List of [r, g, b] values in [0, 1]")
    num_points: int
    depth_base64: str = Field(..., description="Base64-encoded depth colormap PNG")
