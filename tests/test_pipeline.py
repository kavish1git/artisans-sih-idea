"""Automated end-to-end integration tests for ProductImagePipeline (Module 11 & 12)."""

from pathlib import Path
import pytest
from PIL import Image

from src.pipeline import process_product_image, ProductImagePipeline
from src.utils import ImageValidationError

TEST_IMAGES_DIR = Path(__file__).parent / "images"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


class TestFullPipeline:
    """Integration test suite executing full CV pipeline on test fixtures."""

    def test_end_to_end_pottery_processing(self):
        img_path = TEST_IMAGES_DIR / "pottery_normal.jpg"
        result = process_product_image(
            image_path=img_path,
            background="white",
            aspect_ratio="1:1",
            enhancement="auto",
            shadow_mode="professional",
            output_format="JPEG",
            target_dim=1080,
        )

        assert result["status"] == "success"
        assert result["output_image"] is not None
        assert result["transparent_image"] is not None
        assert result["processing_time_ms"] > 0
        assert len(result["dominant_colors"]) >= 1

        # Validate generated catalog image file
        out_file = Path(result["output_image"])
        assert out_file.exists()
        with Image.open(out_file) as img:
            assert img.size == (1080, 1080)
            assert img.mode == "RGB"

        # Validate generated transparent cutout file
        trans_file = Path(result["transparent_image"])
        assert trans_file.exists()
        with Image.open(trans_file) as t_img:
            assert t_img.mode == "RGBA"

        # Quality metrics check
        assert "before_score" in result
        assert "after_score" in result
        assert result["after_score"] >= 80

        # Clean up generated artifacts
        if out_file.exists():
            out_file.unlink()
        if trans_file.exists():
            trans_file.unlink()

    def test_end_to_end_dark_textile_processing(self):
        img_path = TEST_IMAGES_DIR / "textile_dark.jpg"
        result = process_product_image(
            image_path=img_path,
            background="off-white",
            aspect_ratio="1:1",
            enhancement="auto",
            skip_quality_gate=True,
        )

        assert result["status"] == "success"
        # After lighting correction and clean backdrop, quality should improve
        assert result["after_score"] > result["before_score"]
        assert result["improvement"] > 0

        # Clean up
        if result["output_image"] and Path(result["output_image"]).exists():
            Path(result["output_image"]).unlink()
        if result["transparent_image"] and Path(result["transparent_image"]).exists():
            Path(result["transparent_image"]).unlink()

    def test_corrupt_file_handling(self):
        corrupt_path = TEST_IMAGES_DIR / "corrupted.jpg"
        # When quality gate is active, pipeline catches corrupt file and returns needs_retake structured decision
        gate_res = process_product_image(corrupt_path)
        assert gate_res["status"] == "needs_retake"
        assert gate_res["accepted"] is False
        assert any(i["code"] == "IMAGE_CORRUPTED" for i in gate_res["issues"])

        # When bypassing quality gate, underlying low-level decoder raises ImageValidationError
        with pytest.raises(ImageValidationError):
            process_product_image(corrupt_path, skip_quality_gate=True)

