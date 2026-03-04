"""Tests for Pydantic request/response schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from depthview.schemas.depth import (
    DepthRequest,
    DepthResponse,
    HealthResponse,
    PointCloudRequest,
    PointCloudResponse,
)


class TestHealthResponse:
    def test_defaults(self):
        resp = HealthResponse(model="test", device="cpu")
        assert resp.status == "ok"


class TestDepthRequest:
    def test_valid(self):
        req = DepthRequest(image_base64="abc123")
        assert req.fov_deg == 60.0

    def test_fov_bounds(self):
        with pytest.raises(ValidationError):
            DepthRequest(image_base64="abc", fov_deg=5.0)
        with pytest.raises(ValidationError):
            DepthRequest(image_base64="abc", fov_deg=180.0)


class TestPointCloudRequest:
    def test_defaults(self):
        req = PointCloudRequest(image_base64="abc")
        assert req.max_points == 15000
        assert req.colormap == "turbo"

    def test_max_points_bounds(self):
        with pytest.raises(ValidationError):
            PointCloudRequest(image_base64="abc", max_points=50)
        with pytest.raises(ValidationError):
            PointCloudRequest(image_base64="abc", max_points=200000)


class TestDepthResponse:
    def test_valid(self):
        resp = DepthResponse(depth_base64="abc", width=640, height=480)
        assert resp.width == 640


class TestPointCloudResponse:
    def test_valid(self):
        resp = PointCloudResponse(
            points=[[1.0, 2.0, 3.0]],
            colors=[[0.5, 0.5, 0.5]],
            num_points=1,
            depth_base64="abc",
        )
        assert resp.num_points == 1
