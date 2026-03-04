"""Tests for the FastAPI application with a mocked depth estimator."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from depthview.serving.api import create_app


def make_test_image_base64() -> str:
    """Create a small test image and return its base64 encoding."""
    img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.fixture
def client(mock_estimator):
    app = create_app(mock_estimator)
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == "mock-model"


class TestDepthEndpoint:
    def test_valid_image(self, client):
        resp = client.post("/depth", json={"image_base64": make_test_image_base64()})
        assert resp.status_code == 200
        data = resp.json()
        assert "depth_base64" in data
        assert data["width"] == 32
        assert data["height"] == 32

    def test_invalid_image(self, client):
        resp = client.post("/depth", json={"image_base64": "not_valid_base64!!!"})
        assert resp.status_code == 400


class TestPointCloudEndpoint:
    def test_valid_request(self, client):
        resp = client.post(
            "/pointcloud",
            json={"image_base64": make_test_image_base64(), "max_points": 500},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_points"] <= 500
        assert len(data["points"]) == data["num_points"]
        assert len(data["colors"]) == data["num_points"]

    def test_custom_fov(self, client):
        resp = client.post(
            "/pointcloud",
            json={"image_base64": make_test_image_base64(), "fov_deg": 90.0, "max_points": 100},
        )
        assert resp.status_code == 200

    def test_invalid_fov(self, client):
        resp = client.post(
            "/pointcloud",
            json={"image_base64": make_test_image_base64(), "fov_deg": 5.0},
        )
        assert resp.status_code == 422
