"""Automated test suite for Step 2: Image Loading, Validation, and Quality Analysis."""

from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from src.utils import (
    load_image_safely,
    validate_image_bytes,
    ImageValidationError,
    generate_unique_filename,
)
from src.quality import ImageQualityAnalyzer, analyze_image_quality

TEST_IMAGES_DIR = Path(__file__).parent / "images"


class TestImageLoadingAndValidation:
    """Tests secure file intake, MIME detection, size and dimension validation."""

    def test_load_valid_image(self):
        img_path = TEST_IMAGES_DIR / "pottery_normal.jpg"
        bgr, rgb_pil = load_image_safely(img_path)
        assert isinstance(bgr, np.ndarray)
        assert isinstance(rgb_pil, Image.Image)
        assert bgr.shape[0] == 1200
        assert bgr.shape[1] == 1200
        assert bgr.shape[2] == 3

    def test_reject_corrupt_image(self):
        corrupt_path = TEST_IMAGES_DIR / "corrupted.jpg"
        with pytest.raises(ImageValidationError) as exc_info:
            load_image_safely(corrupt_path)
        assert "Unsupported or corrupted file format" in str(exc_info.value)

    def test_reject_tiny_resolution(self):
        tiny_path = TEST_IMAGES_DIR / "tiny_res.jpg"
        with pytest.raises(ImageValidationError) as exc_info:
            load_image_safely(tiny_path)
        assert "too small" in str(exc_info.value)

    def test_reject_empty_bytes(self):
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image_bytes(b"")
        assert "empty" in str(exc_info.value)

    def test_reject_oversized_file(self):
        # 16 MB dummy payload
        oversized = b"\xff\xd8\xff" + b"\x00" * (16 * 1024 * 1024)
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image_bytes(oversized)
        assert "exceeds maximum allowed limit" in str(exc_info.value)

    def test_generate_unique_filename(self):
        fn1 = generate_unique_filename(".jpg")
        fn2 = generate_unique_filename("png")
        assert fn1 != fn2
        assert fn1.endswith(".jpg")
        assert fn2.endswith(".png")


class TestImageQualityAnalyzer:
    """Tests Module 1 quality evaluation and Module 21 artisan accessibility voice feedback."""

    def test_normal_pottery_quality(self):
        img_path = TEST_IMAGES_DIR / "pottery_normal.jpg"
        report = analyze_image_quality(img_path)

        # Check required schema keys
        assert "resolution" in report
        assert "blur" in report
        assert "brightness" in report
        assert "exposure" in report
        assert "noise" in report
        assert "composition" in report
        assert "overall_score" in report
        assert "recommendation" in report

        # Pottery normal should achieve a high quality score
        assert report["overall_score"] >= 80
        assert report["blur"]["blur_status"] == "good"
        assert report["brightness"]["status"] == "good"
        assert report["exposure"]["status"] == "balanced"
        assert "great" in report["recommendation"].lower() or "clearly" in report["recommendation"].lower()

    def test_blur_detection(self):
        img_path = TEST_IMAGES_DIR / "pottery_blurry.jpg"
        report = analyze_image_quality(img_path)

        assert report["blur"]["blur_status"] in ("warning", "bad")
        assert report["overall_score"] < 80
        assert "blurry" in report["recommendation"].lower() or "steady" in report["recommendation"].lower()

    def test_dark_underexposure_detection(self):
        img_path = TEST_IMAGES_DIR / "textile_dark.jpg"
        report = analyze_image_quality(img_path)

        assert report["brightness"]["status"] == "too_dark" or report["exposure"]["status"] == "severe_underexposure"
        assert report["overall_score"] < 70
        assert "dark" in report["recommendation"].lower() or "light" in report["recommendation"].lower()

    def test_bright_overexposure_detection(self):
        img_path = TEST_IMAGES_DIR / "handicraft_bright.jpg"
        report = analyze_image_quality(img_path)

        assert report["brightness"]["status"] == "too_bright" or report["exposure"]["status"] == "severe_overexposure"
        assert "glare" in report["recommendation"].lower() or "sunlight" in report["recommendation"].lower() or "light" in report["recommendation"].lower()

    def test_small_product_composition(self):
        img_path = TEST_IMAGES_DIR / "craft_small.jpg"
        report = analyze_image_quality(img_path)

        assert report["composition"]["status"] == "too_small"
        assert "closer" in report["recommendation"].lower()

    def test_resolution_metrics(self):
        img_path = TEST_IMAGES_DIR / "pottery_normal.jpg"
        report = analyze_image_quality(img_path)

        res = report["resolution"]
        assert res["width"] == 1200
        assert res["height"] == 1200
        assert res["megapixels"] == 1.44
        assert res["status"] == "good"
