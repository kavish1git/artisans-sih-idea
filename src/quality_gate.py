"""Intelligent Image Quality & Identifiability Gate (Module 1, 14, 21).
Pre-flight gate preventing blind catalog processing by verifying:
1. File integrity & decode validity
2. Spatial resolution
3. Sharpness & blur (Laplacian variance & edge analysis)
4. Brightness & exposure (too dark, overexposed, harsh glare)
5. Product visibility & occupancy (product too small, mostly background)
6. Product framing (product cut off at image boundaries)
7. Clutter & multiple competing objects
8. Main product detection
9. Craft identification confidence (prevents low-confidence hallucinations)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from src.utils import load_image_safely
from src.classification import get_classifier


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "quality_thresholds.yaml"

# Safe fallback thresholds if configuration file is missing
DEFAULT_THRESHOLDS = {
    "blur_threshold_variance": 45.0,
    "blur_score_min": 40,
    "brightness_min": 50.0,
    "brightness_max": 225.0,
    "underexposed_pixel_pct": 0.25,
    "overexposed_pixel_pct": 0.20,
    "minimum_width": 400,
    "minimum_height": 400,
    "minimum_product_area_pct": 10.0,
    "maximum_product_area_pct": 95.0,
    "border_cutoff_margin_px": 4,
    "max_clutter_ratio": 0.70,
    "minimum_detection_confidence": 0.50,
    "minimum_classification_confidence": 0.50,
}


def load_quality_thresholds(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads quality gate thresholds from YAML config file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    thresholds = dict(DEFAULT_THRESHOLDS)

    if path.exists() and _HAS_YAML:
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    thresholds.update(loaded)
        except Exception:
            pass

    return thresholds


class QualityGate:
    """Pre-flight quality gate evaluating image usability before downstream processing."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.thresholds = load_quality_thresholds(config_path)
        self.classifier = get_classifier()

    def evaluate(
        self,
        source: Union[str, Path, bytes, np.ndarray, Image.Image],
    ) -> Dict[str, Any]:
        """
        Runs comprehensive pre-flight quality and identifiability gate.
        Returns structured decision dictionary.
        """
        # 1. Image Decode & File Integrity Check
        try:
            if isinstance(source, np.ndarray):
                bgr_img = source.copy()
                if len(bgr_img.shape) == 2:
                    bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_GRAY2BGR)
                rgb_pil = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
            elif isinstance(source, Image.Image):
                rgb_pil = source.convert("RGB")
                bgr_img = cv2.cvtColor(np.array(rgb_pil), cv2.COLOR_RGB2BGR)
            else:
                bgr_img, rgb_pil = load_image_safely(source)
            rgb_img = np.array(rgb_pil)
        except Exception as e:
            return {
                "accepted": False,
                "action": "RETAKE_PHOTO",
                "quality": {
                    "overall_score": 0,
                    "blur_score": 0,
                    "brightness_score": 0,
                    "resolution_score": 0,
                    "framing_score": 0,
                    "product_visibility_score": 0,
                    "background_clutter_score": 0,
                    "detection_confidence": 0.0,
                    "classification_confidence": 0.0,
                },
                "issues": [
                    {
                        "code": "IMAGE_CORRUPTED",
                        "severity": "high",
                        "message": "The photo file could not be read or is corrupted.",
                    }
                ],
                "retake_instruction": "Please take a new photo using your phone's camera.",
                "suggestions": [
                    "Take a new photo using your phone's camera",
                    "Make sure the photo saves completely before uploading",
                ],
                "voice_prompt": "The photo could not be opened. Please take a new photo with your camera.",
            }

        h, w = bgr_img.shape[:2]
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        detected_issues: List[Dict[str, Any]] = []

        # -------------------------------------------------------------
        # 2. Resolution Analysis
        # -------------------------------------------------------------
        min_w = self.thresholds.get("minimum_width", 400)
        min_h = self.thresholds.get("minimum_height", 400)

        if w >= 1080 and h >= 1080:
            res_score = 100
        elif w >= 600 and h >= 600:
            res_score = 80
        elif w >= min_w and h >= min_h:
            res_score = 60
        else:
            res_score = 25

        if w < min_w or h < min_h:
            detected_issues.append({
                "code": "LOW_RESOLUTION",
                "severity": "high",
                "message": "The image quality is too low. Please take a new photo using your phone's normal camera.",
                "tip": "Use your phone's primary camera without zooming",
            })

        # -------------------------------------------------------------
        # 3. Blur & Sharpness Analysis (Laplacian Variance)
        # -------------------------------------------------------------
        std_dev = float(np.std(gray))
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        raw_variance = float(laplacian.var())
        # Contrast-normalized variance
        norm_variance = raw_variance if std_dev > 10.0 else raw_variance * (35.0 / (std_dev + 1e-3))

        blur_thresh = float(self.thresholds.get("blur_threshold_variance", 45.0))
        if norm_variance >= 110.0:
            blur_score = 100
        elif norm_variance >= blur_thresh:
            blur_score = int(50 + (norm_variance - blur_thresh) / (110.0 - blur_thresh) * 50)
        else:
            blur_score = max(5, int((norm_variance / blur_thresh) * 50))

        # Only evaluate blur if image has reasonable dynamic range (not pitch dark or blown out)
        mean_lum_pre = float(np.mean(gray))
        if 35.0 <= mean_lum_pre <= 235.0:
            if norm_variance < blur_thresh or blur_score < self.thresholds.get("blur_score_min", 40):
                # Check for directional motion blur vs uniform defocus
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3).var()
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3).var()
                ratio = max(sobel_x, sobel_y) / (min(sobel_x, sobel_y) + 1e-3)

                is_motion = ratio > 2.2 and norm_variance < 30.0
                detected_issues.append({
                    "code": "EXTREME_MOTION_BLUR" if is_motion else "BLUR_TOO_HIGH",
                    "severity": "high",
                    "message": "Your photo is too blurry. Please hold the phone steady and take another photo.",
                    "tip": "Hold the phone steady and tap the screen to focus",
                })

        # -------------------------------------------------------------
        # 4. Brightness & Exposure Analysis
        # -------------------------------------------------------------
        mean_lum = float(np.mean(gray))
        center_y1, center_y2 = int(h * 0.15), int(h * 0.85)
        center_x1, center_x2 = int(w * 0.15), int(w * 0.85)
        center_lum = float(np.mean(gray[center_y1:center_y2, center_x1:center_x2]))
        dark_ratio = float(np.mean(gray < 15))
        bright_ratio = float(np.mean(gray > 245))

        b_min = float(self.thresholds.get("brightness_min", 48.0))
        b_max = float(self.thresholds.get("brightness_max", 225.0))
        under_limit = float(self.thresholds.get("underexposed_pixel_pct", 0.45))
        over_limit = float(self.thresholds.get("overexposed_pixel_pct", 0.30))

        if 80.0 <= mean_lum <= 185.0 and bright_ratio < 0.12:
            brightness_score = 100
        elif 50.0 <= mean_lum <= 215.0 and bright_ratio < over_limit:
            brightness_score = 75
        else:
            dist = min(abs(mean_lum - b_min), abs(mean_lum - b_max))
            brightness_score = max(10, int(50 - (dist * 0.4)))

        # Genuinely dark: both overall and center are dark, or overwhelming black crush
        if (mean_lum < b_min and center_lum < 52.0) or (dark_ratio > under_limit and center_lum < 45.0):
            detected_issues.append({
                "code": "IMAGE_TOO_DARK",
                "severity": "high",
                "message": "The photo is too dark. Please move to a brighter place or use natural light.",
                "tip": "Take the photo near a window or in daylight",
            })
        elif mean_lum > b_max or bright_ratio > over_limit:
            detected_issues.append({
                "code": "IMAGE_OVEREXPOSED",
                "severity": "high",
                "message": "The photo is too bright. Please avoid direct strong light and take the photo again.",
                "tip": "Avoid harsh flash or direct glare on the craft",
            })

        # -------------------------------------------------------------
        # 5. Product Salience, Visibility & Framing
        # -------------------------------------------------------------
        # Morphological gradient & edge thresholding to locate foreground object
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = float(w * h)

        # Filter meaningful contours (> 0.5% area)
        valid_contours = [c for c in contours if cv2.contourArea(c) > (total_area * 0.005)]

        product_detected = False
        product_area_pct = 0.0
        framing_score = 75
        clutter_score = 80
        prod_box: Optional[Tuple[int, int, int, int]] = None

        if not valid_contours:
            # Completely flat, blank or featureless image
            product_visibility_score = 10
            framing_score = 20
            detected_issues.append({
                "code": "PRODUCT_NOT_DETECTED",
                "severity": "high",
                "message": "We couldn't clearly see the product in this photo. Please place the product clearly in front of the camera and take another photo.",
                "tip": "Place the product in the center of the camera view",
            })
        else:
            product_detected = True
            # Largest contour as main product candidate
            largest_c = max(valid_contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(largest_c)
            prod_box = (bx, by, bw, bh)
            product_area_pct = round(float((bw * bh) / total_area) * 100.0, 1)

            min_area_pct = float(self.thresholds.get("minimum_product_area_pct", 10.0))
            max_area_pct = float(self.thresholds.get("maximum_product_area_pct", 95.0))
            edge_margin = int(self.thresholds.get("border_cutoff_margin_px", 4))

            # Visibility score
            if 20.0 <= product_area_pct <= 85.0:
                product_visibility_score = 100
            elif 10.0 <= product_area_pct < 20.0:
                product_visibility_score = 75
            else:
                product_visibility_score = max(15, int(product_area_pct * 2.0))

            # Framing score
            touches_left = (bx <= edge_margin)
            touches_right = (bx + bw >= w - edge_margin)
            touches_top = (by <= edge_margin)
            touches_bottom = (by + bh >= h - edge_margin)
            touches_count = sum([touches_left, touches_right, touches_top, touches_bottom])

            if touches_count >= 3:
                framing_score = 35
            elif touches_count >= 1:
                framing_score = 65
            else:
                framing_score = 95

            # Issue 5.1: Product too small
            if product_area_pct < min_area_pct:
                detected_issues.append({
                    "code": "PRODUCT_TOO_SMALL",
                    "severity": "high",
                    "message": "The product is too far away. Please move closer so the full product fills more of the frame.",
                    "tip": "Move the camera closer to the craft",
                })

            # Issue 5.2: Product cut off at borders
            if product_area_pct > max_area_pct or (touches_count >= 3 and product_area_pct > 96.0):
                detected_issues.append({
                    "code": "PRODUCT_CUT_OFF",
                    "severity": "high",
                    "message": "Part of the product is outside the photo. Please make sure the entire product is visible.",
                    "tip": "Step back slightly so all edges of the craft fit inside the photo",
                })

            # Issue 5.3: Multiple competing objects
            large_contours = [c for c in valid_contours if cv2.contourArea(c) > (total_area * 0.08)]
            if len(large_contours) >= 3:
                detected_issues.append({
                    "code": "MULTIPLE_OBJECTS",
                    "severity": "medium",
                    "message": "We found several objects in the photo. Please photograph only the product you want to sell.",
                    "tip": "Remove other items and keep only one product in frame",
                })

            # Issue 5.4: Background clutter
            # Compute gradient energy outside the product bounding box
            mask_out = np.ones((h, w), dtype=bool)
            mask_out[by:by + bh, bx:bx + bw] = False
            if np.sum(mask_out) > 50:
                outside_edges = float(np.mean(gradient[mask_out]))
                inside_edges = float(np.mean(gradient[by:by + bh, bx:bx + bw]))
                if outside_edges > (inside_edges * 1.35) and outside_edges > 45.0:
                    clutter_score = 40
                    detected_issues.append({
                        "code": "BACKGROUND_CLUTTER",
                        "severity": "medium",
                        "message": "The background is making it difficult to see the product. Please place the product on a clear, simple background.",
                        "tip": "Place the craft on a plain cloth, floor, or neutral table",
                    })

        # -------------------------------------------------------------
        # 6. Craft Identification Pre-Check
        # -------------------------------------------------------------
        detection_confidence = 0.90 if product_detected else 0.0
        classification_confidence = 0.0
        category_name = "unknown"
        object_display_name = "Unknown"

        if product_detected and prod_box:
            bx, by, bw, bh = prod_box
            crop_rgb = rgb_img[by:by + bh, bx:bx + bw]
            if crop_rgb.size > 0 and crop_rgb.shape[0] >= 10 and crop_rgb.shape[1] >= 10:
                clf_result = self.classifier.classify_crop(crop_rgb)
                classification_confidence = float(clf_result.get("confidence", 0.0))
                category_name = clf_result.get("category", "unknown")
                object_display_name = clf_result.get("object_name", "Handicraft Item")

                min_clf_conf = float(self.thresholds.get("minimum_classification_confidence", 0.50))
                if classification_confidence < min_clf_conf or category_name.lower() in ("unknown", "unknown craft"):
                    # Only flag low confidence if not already flagged as blurry/dark/small
                    has_visual_issue = any(i["code"] in ("BLUR_TOO_HIGH", "IMAGE_TOO_DARK", "PRODUCT_TOO_SMALL") for i in detected_issues)
                    if not has_visual_issue:
                        detected_issues.append({
                            "code": "LOW_CONFIDENCE",
                            "severity": "high",
                            "message": "We couldn't identify the product clearly. Please take a clearer photo showing the complete product.",
                            "tip": "Show the product from a front or 3/4 angle with good lighting",
                        })

        # -------------------------------------------------------------
        # 7. Multi-Issue Ranking & Actionable Artisan Advice
        # -------------------------------------------------------------
        # Sort issues: 'high' severity first, then by domain code priority
        severity_order = {"high": 0, "medium": 1, "low": 2}
        code_priority = {
            "IMAGE_CORRUPTED": 0,
            "LOW_RESOLUTION": 1,
            "IMAGE_TOO_DARK": 2,
            "IMAGE_OVEREXPOSED": 2,
            "PRODUCT_NOT_DETECTED": 3,
            "PRODUCT_TOO_SMALL": 4,
            "PRODUCT_CUT_OFF": 5,
            "BLUR_TOO_HIGH": 6,
            "EXTREME_MOTION_BLUR": 6,
            "LOW_CONFIDENCE": 7,
            "MULTIPLE_OBJECTS": 8,
            "BACKGROUND_CLUTTER": 9,
        }
        detected_issues.sort(key=lambda x: (
            severity_order.get(x.get("severity", "low"), 2),
            code_priority.get(x.get("code", ""), 10),
        ))
        top_issues = detected_issues[:3]

        has_high_issue = any(i.get("severity") == "high" for i in top_issues)
        has_multiple_issues = len(top_issues) >= 2
        accepted = not (has_high_issue or has_multiple_issues)
        action = "PROCESS_IMAGE" if accepted else "RETAKE_PHOTO"

        # Formulate human-friendly retake instruction
        if not accepted and top_issues:
            tips = [i["tip"] for i in top_issues if "tip" in i]
            retake_instruction = " and ".join(tips[:2]).capitalize() + "."
            spoken_prompt = top_issues[0]["message"] + " " + retake_instruction
            suggestions = [i["tip"] for i in top_issues if "tip" in i]
            # Ensure at least 3 helpful tips
            default_tips = [
                "Hold the phone steady and tap to focus",
                "Move closer so the product fills the frame",
                "Use bright, natural daylight without harsh glare",
            ]
            for dt in default_tips:
                if dt not in suggestions and len(suggestions) < 3:
                    suggestions.append(dt)
        else:
            retake_instruction = "Photo is clear and ready for cataloging."
            spoken_prompt = f"{object_display_name} detected clearly! Ready to create catalog listing."
            suggestions = []

        # -------------------------------------------------------------
        # 8. Normalized Scores (0-100)
        # -------------------------------------------------------------
        # Weighted overall quality score
        overall_score = round(
            0.35 * blur_score +
            0.25 * brightness_score +
            0.20 * product_visibility_score +
            0.10 * framing_score +
            0.10 * res_score
        )
        if not accepted:
            overall_score = min(overall_score, 45)

        quality_metrics = {
            "overall_score": int(overall_score),
            "blur_score": int(blur_score),
            "brightness_score": int(brightness_score),
            "resolution_score": int(res_score),
            "framing_score": int(framing_score),
            "product_visibility_score": int(product_visibility_score),
            "background_clutter_score": int(clutter_score),
            "product_area_percentage": float(product_area_pct),
            "detection_confidence": round(float(detection_confidence), 2),
            "classification_confidence": round(float(classification_confidence), 2),
        }

        return {
            "accepted": bool(accepted),
            "action": str(action),
            "quality": quality_metrics,
            "issues": [
                {"code": i["code"], "severity": i["severity"], "message": i["message"]}
                for i in top_issues
            ],
            "retake_instruction": str(retake_instruction),
            "suggestions": suggestions,
            "voice_prompt": str(spoken_prompt),
            "product_detection": {
                "detected": bool(product_detected),
                "name": str(object_display_name),
                "category": str(category_name),
                "confidence": round(float(classification_confidence), 2),
            },
        }


# Global singleton quality gate
_default_gate: Optional[QualityGate] = None


def get_quality_gate(config_path: Optional[Union[str, Path]] = None) -> QualityGate:
    """Returns singleton instance of QualityGate."""
    global _default_gate
    if _default_gate is None or config_path is not None:
        _default_gate = QualityGate(config_path=config_path)
    return _default_gate


def evaluate_image_gate(
    source: Union[str, Path, bytes, np.ndarray, Image.Image],
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Functional convenience helper to evaluate image against quality gate."""
    gate = get_quality_gate(config_path=config_path)
    return gate.evaluate(source)
