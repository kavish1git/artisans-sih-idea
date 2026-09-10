"""Automated test suite for Modules 4 through 10:
Shadows, Lighting, White Balance, Cropping, Background Generation, Enhancement, and Formatting.
"""

from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from src.crop import crop_and_center_product, ProductCropper
from src.lighting import correct_lighting, LightingCorrector
from src.white_balance import correct_white_balance, WhiteBalanceCorrector
from src.shadow import apply_grounding_shadow, ShadowHandler
from src.marketplace import generate_marketplace_image, MarketplaceGenerator
from src.formatter import format_marketplace_image, MarketplaceFormatter

TEST_IMAGES_DIR = Path(__file__).parent / "images"


class TestProcessingModules:
    """Validates individual visual processing modules."""

    @pytest.fixture
    def sample_rgba_image(self):
        """Creates an RGBA test image with a centered circle."""
        img = Image.new("RGBA", (500, 500), (0, 0, 0, 0))
        draw_arr = np.array(img)
        # Add a 200x200 red square in center
        draw_arr[150:350, 150:350] = [200, 40, 40, 255]
        return Image.fromarray(draw_arr, mode="RGBA")

    def test_product_cropping_and_centering(self, sample_rgba_image):
        # Crop with 1:1 aspect ratio and 10% padding
        cropped, bbox = crop_and_center_product(sample_rgba_image, aspect_ratio="1:1", padding=0.10)
        assert isinstance(cropped, Image.Image)
        w, h = cropped.size
        # 1:1 aspect ratio requires w == h
        assert w == h
        assert bbox["width"] == 200
        assert bbox["height"] == 200
        # Centering check: x and y offset should be identical for symmetric square in 1:1
        assert abs(bbox["x"] - bbox["y"]) <= 2

    def test_aspect_ratios(self, sample_rgba_image):
        for ratio_name, (rw, rh) in [("4:5", (4, 5)), ("16:9", (16, 9)), ("3:4", (3, 4))]:
            cropped, bbox = crop_and_center_product(sample_rgba_image, aspect_ratio=ratio_name)
            w, h = cropped.size
            expected_ratio = rw / rh
            actual_ratio = w / h
            assert abs(actual_ratio - expected_ratio) < 0.05

    def test_lighting_correction(self):
        dark_path = TEST_IMAGES_DIR / "textile_dark.jpg"
        img = Image.open(dark_path)
        corrected = correct_lighting(img, intensity=1.0)
        assert isinstance(corrected, Image.Image)

        # Corrected image should be significantly brighter
        lum_orig = np.mean(np.array(img.convert("L")))
        lum_corr = np.mean(np.array(corrected.convert("L")))
        assert lum_corr > lum_orig

    def test_white_balance_correction(self):
        # Create an image with severe yellow/warm indoor tint
        tinted = np.full((300, 300, 3), (255, 230, 110), dtype=np.uint8)
        img = Image.fromarray(tinted)
        wb_corrected = correct_white_balance(img, strength=0.9)
        corr_arr = np.array(wb_corrected)

        # Blue channel should be boosted relative to yellow
        assert corr_arr[:, :, 2].mean() > tinted[:, :, 2].mean()

    def test_grounding_shadow_modes(self, sample_rgba_image):
        prof_shadow = apply_grounding_shadow(sample_rgba_image, mode="professional")
        nat_shadow = apply_grounding_shadow(sample_rgba_image, mode="natural")

        assert isinstance(prof_shadow, Image.Image)
        assert isinstance(nat_shadow, Image.Image)
        assert prof_shadow.mode == "RGBA"
        assert nat_shadow.mode == "RGBA"

        # Check shadow presence in alpha layer outside product area
        alpha_prof = np.array(prof_shadow.split()[-1])
        assert np.any(alpha_prof[355:400, :] > 0)

    def test_marketplace_background_generation(self, sample_rgba_image):
        # Test white background
        white_bg = generate_marketplace_image(sample_rgba_image, background="white")
        assert white_bg.mode == "RGB"
        # Background pixel at top left (0,0) must be white (255, 255, 255)
        top_left = white_bg.getpixel((0, 0))
        assert top_left == (255, 255, 255)

        # Test off-white background
        offwhite_bg = generate_marketplace_image(sample_rgba_image, background="off-white")
        assert offwhite_bg.getpixel((0, 0)) == (248, 249, 250)

    def test_marketplace_formatter(self, sample_rgba_image):
        # Format to default 1080x1080 JPEG
        formatted = format_marketplace_image(sample_rgba_image, width=1080, height=1080, export_format="JPEG")
        assert formatted.size == (1080, 1080)
        assert formatted.mode == "RGB"

        # Test export bytes
        jpeg_bytes = MarketplaceFormatter.export_bytes(formatted, export_format="JPEG", quality=90)
        assert len(jpeg_bytes) > 0
        assert jpeg_bytes.startswith(b"\xff\xd8\xff")  # Valid JPEG magic bytes
