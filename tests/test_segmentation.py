"""Automated test suite for Step 3 & 4: Product Segmentation and Background Removal."""

from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from src.segmentation import ProductSegmenter, segment_product, SegmentationResult
from src.background import BackgroundRemover, remove_background

TEST_IMAGES_DIR = Path(__file__).parent / "images"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


class TestProductSegmentation:
    """Tests Module 2 and Module 3 segmentation and background removal."""

    def test_segment_pottery_image(self):
        img_path = TEST_IMAGES_DIR / "pottery_normal.jpg"
        result = segment_product(img_path)

        assert isinstance(result, SegmentationResult)
        assert isinstance(result.mask, np.ndarray)
        assert result.mask.ndim == 2
        assert result.mask.dtype == np.uint8
        assert result.rgba_foreground.shape[2] == 4

        # Bounding box should encompass the pot
        bbox = result.bounding_box
        assert bbox["width"] > 200
        assert bbox["height"] > 200
        assert result.confidence > 0.4
        assert result.model_used in ("u2net", "grabcut_saliency")

    def test_remove_background_rgba(self):
        img_path = TEST_IMAGES_DIR / "pottery_normal.jpg"
        transparent_img, seg_result = remove_background(img_path)

        assert isinstance(transparent_img, Image.Image)
        assert transparent_img.mode == "RGBA"
        assert transparent_img.size == (1200, 1200)

        # Alpha channel should have both 0 (background) and 255 (foreground)
        alpha_channel = np.array(transparent_img.split()[-1])
        assert np.any(alpha_channel == 0)
        assert np.any(alpha_channel > 200)

    def test_save_transparent_png(self):
        img_path = TEST_IMAGES_DIR / "pottery_normal.jpg"
        out_path = OUTPUT_DIR / "test_transparent.png"
        remover = BackgroundRemover()
        seg_res = remover.save_transparent_png(img_path, out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 1000

        # Verify saved PNG
        with Image.open(out_path) as loaded:
            assert loaded.mode == "RGBA"
        # Cleanup
        if out_path.exists():
            out_path.unlink()
