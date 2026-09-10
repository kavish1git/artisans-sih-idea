"""Automated tests for Fully Automatic AI Pipeline & New Components.
Tests:
- Automatic Indian handicraft detection & confidence calibration
- Objective visual attributes extraction
- End-to-end single-image processing with zero manual controls
- API endpoint accepting ONLY image file
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.pipeline import process_product_image
from src.detector import detect_product
from src.attributes import extract_attributes
from src.segmentation import segment_product

client = TestClient(app)
TEST_IMAGES_DIR = Path(__file__).parent / "images"


class TestAutomaticPipeline:
    """Verifies automatic detection, attributes, and zero-configuration flow."""

    def test_indian_handicraft_detection_pottery(self):
        img_path = TEST_IMAGES_DIR / "pottery_normal.jpg"
        seg = segment_product(img_path)
        det = detect_product(seg.rgba_foreground[:, :, :3], mask=seg.mask, bounding_box=seg.bounding_box)

        assert det["detected"] is True
        assert det["category"] == "Pottery"
        assert "Pottery" in det["name"] or "Ceramic" in det["name"]
        assert det["confidence"] >= 0.85
        assert det["confidence_label"] == "detected"
        assert "Pottery" in det["display_text"] or "Ceramic" in det["display_text"]

    def test_visual_attributes_extraction(self):
        img_path = TEST_IMAGES_DIR / "pottery_normal.jpg"
        seg = segment_product(img_path)
        attrs = extract_attributes(seg.rgba_foreground[:, :, :3], mask=seg.mask, bounding_box=seg.bounding_box)

        assert "dominant_colors" in attrs
        assert len(attrs["dominant_colors"]) >= 1
        assert "palette_hex" in attrs
        assert len(attrs["palette_hex"]) >= 1
        assert attrs["shape"] in ("round", "rectangular", "oval", "square", "elongated", "irregular")
        assert attrs["orientation"] in ("upright", "horizontal", "square")
        assert attrs["pattern"] in ("patterned", "plain", "subtly textured")
        assert attrs["texture"] in ("smooth / glazed", "woven-looking", "carved / granular", "textured", "metallic / polished")

    def test_single_image_pipeline_zero_controls(self):
        img_path = TEST_IMAGES_DIR / "pottery_normal.jpg"
        # Call with ONLY the image path, no manual controls
        result = process_product_image(img_path)

        assert result["status"] == "success"
        # Schema verification matching Sections 13 & 19
        assert "product" in result
        assert result["product"]["category"] == "Pottery"
        assert result["product"]["confidence"] > 0.60
        assert "visual_attributes" in result
        assert "geometry" in result
        assert "image" in result
        assert result["image"]["width"] == 1080
        assert result["image"]["height"] == 1080
        assert "quality" in result
        assert result["quality"]["after"] >= 80
        assert "processing" in result
        assert "voice_prompt" in result
        assert len(result["voice_prompt"]) > 10

    def test_api_single_image_upload_only(self):
        img_path = TEST_IMAGES_DIR / "pottery_normal.jpg"
        with open(img_path, "rb") as f:
            # POST with ONLY image in multipart form-data (no other fields)
            response = client.post(
                "/api/v1/process-image",
                files={"image": ("pottery.jpg", f, "image/jpeg")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "product" in data
        assert data["product"]["category"] == "Pottery"
        assert "visual_attributes" in data
        assert "colors" in data["visual_attributes"]
        assert "shape" in data["visual_attributes"]
        assert "image" in data
        assert data["image"]["processed"].startswith("http")
        assert "voice_prompt" in data
