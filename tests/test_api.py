"""Automated tests for FastAPI endpoints (Module 13)."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
TEST_IMAGES_DIR = Path(__file__).parent / "images"


class TestAPIEndpoints:
    """Verifies FastAPI REST endpoints."""

    def test_health_check(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_root_demo_ui(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "Artisan Smart Cataloging" in response.text

    def test_quality_check_endpoint(self):
        img_path = TEST_IMAGES_DIR / "pottery_normal.jpg"
        with open(img_path, "rb") as f:
            response = client.post(
                "/api/v1/quality",
                files={"image": ("pottery.jpg", f, "image/jpeg")},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "overall_score" in data
        assert "recommendation" in data
        assert data["overall_score"] >= 80

    def test_process_image_endpoint(self):
        img_path = TEST_IMAGES_DIR / "pottery_normal.jpg"
        with open(img_path, "rb") as f:
            response = client.post(
                "/api/v1/process-image",
                files={"image": ("pottery.jpg", f, "image/jpeg")},
                data={
                    "background": "white",
                    "aspect_ratio": "1:1",
                    "enhancement": "auto",
                    "shadow_mode": "professional",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "image_url" in data
        assert "transparent_url" in data
        assert "before_score" in data
        assert "after_score" in data
        assert "dominant_colors" in data
        assert len(data["dominant_colors"]) > 0
        assert data["processing_time_ms"] > 0

    def test_reject_corrupted_upload(self):
        corrupt_path = TEST_IMAGES_DIR / "corrupted.jpg"
        with open(corrupt_path, "rb") as f:
            response = client.post(
                "/api/v1/process-image",
                files={"image": ("corrupted.jpg", f, "image/jpeg")},
            )
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
