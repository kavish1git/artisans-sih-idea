"""Regression test: Pottery must NEVER be classified as Phulkari Dupatta (Requirement 4).
Verifies that decorative pottery is recognized in the pottery family
and does not hallucinate textile categories.
"""

from pathlib import Path
import pytest
import cv2

from src.pipeline import process_product_image
from src.classification import classify_product
from src.segmentation import segment_product

TEST_IMAGES_DIR = Path(__file__).parent / "images"


class TestPotteryRegression:
    """Verifies pottery regression test on user's actual decorative pot photograph."""

    def test_pottery_is_not_dupatta(self):
        img_path = TEST_IMAGES_DIR / "user_decorative_pottery.jpg"
        assert img_path.exists(), "User decorative pottery test fixture must exist"

        result = process_product_image(img_path)

        assert result["status"] == "success"
        category = result["product"]["category"]
        name = result["product"]["name"]

        # MUST NOT be Phulkari Dupatta
        assert "dupatta" not in category.lower(), f"Pottery was misidentified as {category}"
        assert "phulkari" not in name.lower(), f"Pottery was misidentified as {name}"

        # MUST be in pottery family
        assert category.lower() in ("pottery", "terracotta_pottery", "home_decor", "other_handicraft") or category in ("Pottery", "Home Decor"), (
            f"Expected pottery-related category, got: {category}"
        )

    def test_calibrated_confidence_not_overconfident(self):
        img_path = TEST_IMAGES_DIR / "user_decorative_pottery.jpg"
        result = process_product_image(img_path)

        # Confidence status should be calibrated, not blindly claiming 0.96
        conf = result["product"]["confidence"]
        assert conf <= 0.95 or result["product"].get("confidence_status") != "high", (
            f"Overconfident prediction: {conf}"
        )
