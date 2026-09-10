"""Automated tests for Full Scene / Textile Stack Composition Preservation.
Ensures that:
1. Multi-part craft compositions and folded textile stacks are NOT sliced into pieces.
2. The entire photo composition is preserved and formatted cleanly to 1080x1080.
3. Category is accurately detected as Textile / Handloom Textile.
4. Clean photos with plain/smooth backgrounds are not falsely flagged as blurry.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image

from src.pipeline import process_product_image
from src.composition import analyze_composition_suitability
from src.quality_gate import evaluate_image_gate

TEST_IMAGES_DIR = Path(__file__).parent / "images"


class TestCompositionPreservation:
    """Verifies non-destructive composition handling on folded fabric stacks and full craft scenes."""

    def test_textile_stack_preserved_as_whole(self):
        img_path = TEST_IMAGES_DIR / "user_textile_hires.png"
        assert img_path.exists()

        result = process_product_image(img_path)

        assert result["status"] == "success"
        assert result["accepted"] is True

        # Product classification should be Textile / Handloom Textile
        product = result["product"]
        assert product["category"] == "Textile"
        assert "textile" in product["name"].lower() or "handloom" in product["name"].lower()
        assert product["confidence"] >= 0.75

        # Output catalog image must be a complete 1080x1080 image
        out_path = Path(result["output_image"])
        assert out_path.exists()
        with Image.open(out_path) as out_img:
            assert out_img.size == (1080, 1080)
            assert out_img.mode == "RGB"

        # Verify that the entire image composition was preserved
        proc = result.get("processing", {})
        assert proc.get("shadow_applied") == "none (full_composition)"

    def test_sharp_craft_on_plain_bg_not_blurry(self):
        """Plain or smooth backgrounds must not cause clear crafts to be flagged as blurry."""
        plain_bg = np.full((1000, 1000, 3), 245, dtype=np.uint8)
        # Sharp patterned pot in center
        cv2.circle(plain_bg, (500, 500), 120, (50, 100, 200), -1)
        for r in range(10, 110, 15):
            cv2.circle(plain_bg, (500, 500), r, (255, 255, 255), 2)

        gate_res = evaluate_image_gate(plain_bg)
        assert gate_res["quality"]["blur_score"] >= 60
        issue_codes = [i["code"] for i in gate_res.get("issues", [])]
        assert "BLUR_TOO_HIGH" not in issue_codes
        assert "EXTREME_MOTION_BLUR" not in issue_codes
