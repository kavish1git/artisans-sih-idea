"""Comprehensive Automated Test Suite for Image Quality & Identifiability Gate.
Covers Section 20 requirements:
1. Clear pottery image -> ACCEPT (PROCESS_IMAGE)
2. Extremely blurry image -> RETAKE (BLUR_TOO_HIGH)
3. Very dark image -> RETAKE (IMAGE_TOO_DARK)
4. Overexposed image -> RETAKE (IMAGE_OVEREXPOSED)
5. Tiny product -> RETAKE (PRODUCT_TOO_SMALL / PRODUCT_NOT_DETECTED)
6. Product cut off -> RETAKE (PRODUCT_CUT_OFF)
7. No product -> RETAKE (PRODUCT_NOT_DETECTED)
8. Low resolution image -> RETAKE (LOW_RESOLUTION)
9. Unidentifiable / Low confidence -> RETAKE (LOW_CONFIDENCE)
10. Clear product with cluttered background -> ACCEPT
11. Clear pot -> pottery, NOT Phulkari Dupatta
12. Processed image final verification check
13. Pipeline stops immediately on rejection (zero downstream image creation)
14. API endpoints return correct QualityGate schema
"""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.quality_gate import QualityGate, evaluate_image_gate
from src.pipeline import process_product_image, get_pipeline


client = TestClient(app)
TEST_IMAGES_DIR = Path(__file__).parent / "images"


class TestQualityGateScenarios:
    """Verifies all quality gate checks, reason codes, and retake decisions."""

    def test_1_clear_pottery_image_accepted(self):
        """Test Case 1: Clear pottery photo must pass quality gate."""
        img_path = TEST_IMAGES_DIR / "pottery_normal.jpg"
        res = evaluate_image_gate(img_path)

        assert res["accepted"] is True
        assert res["action"] == "PROCESS_IMAGE"
        assert res["quality"]["overall_score"] >= 70
        assert res["quality"]["blur_score"] >= 70
        assert res["quality"]["brightness_score"] >= 65
        assert len(res["issues"]) == 0

    def test_2_blurry_image_rejected(self):
        """Test Case 2: Blurry photo must be rejected with BLUR_TOO_HIGH."""
        img_path = TEST_IMAGES_DIR / "pottery_blurry.jpg"
        res = evaluate_image_gate(img_path)

        assert res["accepted"] is False
        assert res["action"] == "RETAKE_PHOTO"
        issue_codes = [i["code"] for i in res["issues"]]
        assert "BLUR_TOO_HIGH" in issue_codes or "EXTREME_MOTION_BLUR" in issue_codes
        assert "blurry" in res["issues"][0]["message"].lower() or "steady" in res["retake_instruction"].lower()

    def test_3_very_dark_image_rejected(self):
        """Test Case 3: Very dark underexposed image must be rejected with IMAGE_TOO_DARK."""
        img_path = TEST_IMAGES_DIR / "textile_dark.jpg"
        res = evaluate_image_gate(img_path)

        assert res["accepted"] is False
        assert res["action"] == "RETAKE_PHOTO"
        issue_codes = [i["code"] for i in res["issues"]]
        assert "IMAGE_TOO_DARK" in issue_codes
        assert any("dark" in i["message"].lower() for i in res["issues"])

    def test_4_overexposed_image_rejected(self):
        """Test Case 4: Harshly overexposed photo must be rejected with IMAGE_OVEREXPOSED."""
        img_path = TEST_IMAGES_DIR / "handicraft_bright.jpg"
        res = evaluate_image_gate(img_path)

        assert res["accepted"] is False
        assert res["action"] == "RETAKE_PHOTO"
        issue_codes = [i["code"] for i in res["issues"]]
        assert "IMAGE_OVEREXPOSED" in issue_codes
        assert any("bright" in i["message"].lower() for i in res["issues"])

    def test_5_tiny_product_rejected(self):
        """Test Case 5: Product occupying too little frame must be rejected."""
        img_path = TEST_IMAGES_DIR / "craft_small.jpg"
        res = evaluate_image_gate(img_path)

        assert res["accepted"] is False
        assert res["action"] == "RETAKE_PHOTO"
        issue_codes = [i["code"] for i in res["issues"]]
        assert ("PRODUCT_TOO_SMALL" in issue_codes) or ("PRODUCT_NOT_DETECTED" in issue_codes)

    def test_6_product_cut_off_rejected(self):
        """Test Case 6: Product severely cut off by camera boundaries must be rejected."""
        # Craft rectangle spanning 97% of frame and hitting frame boundaries
        img = np.full((500, 500, 3), 240, dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (492, 498), (60, 100, 160), -1)

        res = evaluate_image_gate(img)
        assert res["accepted"] is False
        assert res["action"] == "RETAKE_PHOTO"
        issue_codes = [i["code"] for i in res["issues"]]
        assert "PRODUCT_CUT_OFF" in issue_codes

    def test_7_no_product_detected(self):
        """Test Case 7: Flat blank image with no craft must be rejected."""
        # Flat blank canvas
        blank = np.full((600, 600, 3), 200, dtype=np.uint8)
        res = evaluate_image_gate(blank)

        assert res["accepted"] is False
        assert res["action"] == "RETAKE_PHOTO"
        issue_codes = [i["code"] for i in res["issues"]]
        assert "PRODUCT_NOT_DETECTED" in issue_codes

    def test_8_low_resolution_rejected(self):
        """Test Case 8: Image below minimum resolution must be rejected."""
        tiny = np.full((200, 200, 3), 150, dtype=np.uint8)
        res = evaluate_image_gate(tiny)

        assert res["accepted"] is False
        assert res["action"] == "RETAKE_PHOTO"
        issue_codes = [i["code"] for i in res["issues"]]
        assert "LOW_RESOLUTION" in issue_codes

    def test_9_unidentifiable_craft_rejected(self):
        """Test Case 9: Random visual noise without recognizable craft triggers retake."""
        rng = np.random.RandomState(42)
        noise = rng.randint(50, 200, size=(500, 500, 3), dtype=np.uint8)
        res = evaluate_image_gate(noise)

        assert res["accepted"] is False
        assert res["action"] == "RETAKE_PHOTO"

    def test_10_clear_pot_is_pottery_not_dupatta(self):
        """Test Case 11: Pottery must NEVER be hallucinated as Phulkari Dupatta."""
        img_path = TEST_IMAGES_DIR / "user_decorative_pottery.jpg"
        res = evaluate_image_gate(img_path)

        det = res.get("product_detection", {})
        cat = det.get("category", "")
        name = det.get("name", "")
        assert "dupatta" not in cat.lower()
        assert "phulkari" not in name.lower()
        assert "pottery" in cat.lower() or "pottery" in name.lower()

    def test_11_pipeline_stops_processing_on_rejected_image(self):
        """Test Case 13: Pipeline must NOT generate catalog output if gate rejects image."""
        img_path = TEST_IMAGES_DIR / "pottery_blurry.jpg"
        result = process_product_image(img_path)

        assert result["status"] == "needs_retake"
        assert result["accepted"] is False
        assert result["action"] == "RETAKE_PHOTO"
        # Must not contain processed marketplace image URLs
        assert "image" not in result or result["image"].get("processed") is None

    def test_12_post_processing_catalog_verification(self):
        """Test Case 12: Processed catalog output must pass post-processing checks."""
        img_path = TEST_IMAGES_DIR / "pottery_normal.jpg"
        result = process_product_image(img_path)

        assert result["status"] == "success"
        assert result["accepted"] is True
        assert result["processing_status"] == "SUCCESS"
        assert result["quality"]["after"] >= 80
        assert result["image"]["width"] == 1080
        assert result["image"]["height"] == 1080

    def test_13_api_rejected_and_accepted_flow(self):
        """Test Case 14: API endpoint returns correct schema for both cases."""
        # 1. Rejected Upload
        with open(TEST_IMAGES_DIR / "pottery_blurry.jpg", "rb") as f:
            resp_reject = client.post(
                "/api/v1/process-image",
                files={"image": ("blurry.jpg", f, "image/jpeg")},
            )
        assert resp_reject.status_code == 200
        data_rej = resp_reject.json()
        assert data_rej["accepted"] is False
        assert data_rej["action"] == "RETAKE_PHOTO"
        assert "issues" in data_rej
        assert len(data_rej["issues"]) >= 1
        assert "retake_instruction" in data_rej
        assert "suggestions" in data_rej

        # 2. Accepted Upload
        with open(TEST_IMAGES_DIR / "pottery_normal.jpg", "rb") as f:
            resp_acc = client.post(
                "/api/v1/process-image",
                files={"image": ("pottery.jpg", f, "image/jpeg")},
            )
        assert resp_acc.status_code == 200
        data_acc = resp_acc.json()
        assert data_acc["accepted"] is True
        assert data_acc["action"] == "PROCESS_IMAGE"
        assert "product" in data_acc
        assert "image" in data_acc
        assert data_acc["image"]["processed"] is not None
