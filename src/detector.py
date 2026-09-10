"""Automatic Product Detection and Indian Handicraft Recognition Module (Module 2 & 3)
Determines the primary artisan product, classifies Indian handicraft categories,
calibrates prediction confidence, and isolates the primary craft from background clutter.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False


# Comprehensive Indian Handicraft Taxonomy
HANDICRAFT_TAXONOMY = {
    "Textile": {
        "specific_crafts": [
            "Phulkari Dupatta",
            "Banarasi Saree",
            "Pashmina Shawl",
            "Handloom Textile",
            "Embroidered Fabric",
            "Chanderi Craft",
        ],
        "generic_name": "Textile product",
        "imagenet_classes": [834, 783, 614, 804, 407, 815, 915, 682, 721, 608, 655],
        "hues": [(0, 180)],  # Broad color spectrum
    },
    "Pottery": {
        "specific_crafts": [
            "Terracotta Pottery",
            "Clay Earthenware Pot",
            "Ceramic Vase",
            "Glazed Ceramic Craft",
            "Handmade Clay Pitcher",
        ],
        "generic_name": "Pottery craft",
        "imagenet_classes": [605, 710, 844, 499, 968, 659, 898, 899, 504, 849, 950],
        "hues": [(5, 28), (0, 15)],  # Earthy reds, terracottas, warm ochres, or glazed
    },
    "Basket": {
        "specific_crafts": [
            "Handmade Woven Basket",
            "Bamboo Cane Basket",
            "Jute Storage Basket",
            "Coir Craft Basket",
        ],
        "generic_name": "Woven basket",
        "imagenet_classes": [431, 790, 458, 903, 563],
        "hues": [(15, 38)],  # Cane, wicker, bamboo browns & yellows
    },
    "Woodcraft": {
        "specific_crafts": [
            "Hand-Carved Wooden Craft",
            "Sheesham Wood Carving",
            "Bamboo Artifact",
            "Wooden Decorative Figurine",
        ],
        "generic_name": "Wooden craft",
        "imagenet_classes": [919, 514, 403, 875, 470, 770, 807],
        "hues": [(10, 32)],  # Wood tones: cedar, mahogany, teak, sheesham
    },
    "Metalcraft": {
        "specific_crafts": [
            "Brass Idol / Figurine",
            "Bell Metal Craft",
            "Dhokra Metal Art",
            "Engraved Copper Vessel",
        ],
        "generic_name": "Metal craft",
        "imagenet_classes": [427, 468, 572, 498, 594, 466, 444, 883],
        "hues": [(20, 50)],  # Brass gold, copper bronze
    },
    "Jewelry": {
        "specific_crafts": [
            "Handcrafted Earrings",
            "Traditional Necklace",
            "Beaded Kundan Jewelry",
            "Artisan Bangle Set",
            "Meenakari Ornament",
        ],
        "generic_name": "Handmade jewelry",
        "imagenet_classes": [865, 509, 760, 487, 404, 611, 786],
        "hues": [(0, 180)],  # Gold, silver, gems, beads
    },
    "Leather & Bag": {
        "specific_crafts": [
            "Embroidered Handmade Bag",
            "Shantiniketan Leather Craft",
            "Jute Tote Bag",
            "Artisan Leather Pouch",
        ],
        "generic_name": "Handmade bag",
        "imagenet_classes": [748, 637, 414, 803, 477, 658],
        "hues": [(10, 35), (0, 180)],
    },
    "Art & Decor": {
        "specific_crafts": [
            "Traditional Folk Painting",
            "Madhubani / Warli Art",
            "Clay Figurine / Toy",
            "Handmade Candle",
            "Artisan Home Decor",
        ],
        "generic_name": "Decorative craft",
        "imagenet_classes": [470, 840, 643, 917, 875, 722, 608],
        "hues": [(0, 180)],
    },
}


class ProductDetector:
    """Detects and classifies Indian handicrafts using visual features and deep embeddings."""

    def __init__(self, model_path: Optional[str] = None):
        self.session = None
        default_model = Path(__file__).parent.parent / "models" / "mobilenetv2-7.onnx"

        if model_path and os.path.exists(model_path):
            model_target = Path(model_path)
        else:
            model_target = default_model

        if _HAS_ORT and model_target.exists():
            try:
                # Use CPUExecutionProvider for lightweight, cross-platform stability
                self.session = ort.InferenceSession(
                    str(model_target),
                    providers=["CPUExecutionProvider"],
                )
            except Exception:
                self.session = None

    def _preprocess_for_onnx(self, rgb_crop: np.ndarray) -> np.ndarray:
        """Prepares 224x224 normalized tensor for MobileNetV2."""
        resized = cv2.resize(rgb_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
        return arr

    def _extract_visual_cues(self, rgb_crop: np.ndarray, mask_crop: np.ndarray) -> Dict[str, Any]:
        """Extracts domain-specific color, texture, and contour cues."""
        h, w = rgb_crop.shape[:2]
        fg_mask = (mask_crop > 50) if mask_crop is not None else np.ones((h, w), dtype=bool)

        if np.sum(fg_mask) < 20:
            return {"dominant_hue": 0, "aspect_ratio": 1.0, "is_terracotta": False, "is_brass": False}

        fg_rgb = rgb_crop[fg_mask]
        hsv = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2HSV)
        fg_hsv = hsv[fg_mask]

        mean_hue = float(np.median(fg_hsv[:, 0]))
        mean_sat = float(np.mean(fg_hsv[:, 1]))
        mean_val = float(np.mean(fg_hsv[:, 2]))

        # Check for terracotta earthy red/brown tones (H: 8-22, S: > 50, V: 40-220)
        is_terracotta = (8 <= mean_hue <= 24) and (mean_sat > 45) and (mean_val > 40)

        # Check for metallic brass/gold tones (H: 22-45, S: > 40, metallic luster/high val)
        is_brass = (22 <= mean_hue <= 42) and (mean_sat > 40) and (mean_val > 100)

        # High-frequency texture (woven fabric / basketry)
        gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        aspect_ratio = float(w / max(1, h))

        return {
            "mean_hue": mean_hue,
            "mean_sat": mean_sat,
            "mean_val": mean_val,
            "is_terracotta": is_terracotta,
            "is_brass": is_brass,
            "laplacian_var": laplacian_var,
            "aspect_ratio": aspect_ratio,
        }

    def detect_and_classify(
        self,
        rgb_image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        bounding_box: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Main recognition pipeline: determines primary product, classifies craft,
        calibrates confidence, and prevents hallucination.
        """
        h, w = rgb_image.shape[:2]

        # 1. Determine product region
        if bounding_box:
            bx = max(0, bounding_box.get("x", 0))
            by = max(0, bounding_box.get("y", 0))
            bw = min(w - bx, bounding_box.get("width", w))
            bh = min(h - by, bounding_box.get("height", h))
        else:
            bx, by, bw, bh = 0, 0, w, h

        product_crop = rgb_image[by:by + bh, bx:bx + bw]
        mask_crop = mask[by:by + bh, bx:bx + bw] if mask is not None else None

        # Guard against empty/invalid crops
        if product_crop.size == 0 or product_crop.shape[0] < 10 or product_crop.shape[1] < 10:
            return {
                "detected": False,
                "name": "Unknown",
                "category": "Unknown",
                "confidence": 0.0,
                "confidence_label": "not_detected",
                "display_text": "No product detected",
            }

        # 2. Extract visual domain cues
        cues = self._extract_visual_cues(product_crop, mask_crop)

        # 3. Model inference (MobileNetV2 feature logits)
        logits = None
        if self.session is not None:
            try:
                input_tensor = self._preprocess_for_onnx(product_crop)
                ort_inputs = {self.session.get_inputs()[0].name: input_tensor}
                logits = self.session.run(None, ort_inputs)[0][0]
            except Exception:
                logits = None

        # 4. Score each Indian Handicraft category
        category_scores: Dict[str, float] = {}

        for cat_name, cat_info in HANDICRAFT_TAXONOMY.items():
            score = 0.20  # Base prior

            if logits is not None:
                # Sum probabilities of matching ImageNet synsets
                cat_indices = cat_info["imagenet_classes"]
                cat_logits = [logits[idx] for idx in cat_indices if idx < len(logits)]
                if cat_logits:
                    max_logit = max(cat_logits)
                    # Convert to sigmoid-style score [0.0, 1.0]
                    model_score = 1.0 / (1.0 + np.exp(-max_logit / 3.0))
                    score += model_score * 0.60

            # Complement with visual cues
            if cat_name == "Pottery":
                if cues["is_terracotta"]:
                    score += 0.35
                elif cues["aspect_ratio"] > 0.6 and cues["aspect_ratio"] < 1.4:
                    score += 0.10
            elif cat_name == "Metalcraft":
                if cues["is_brass"]:
                    score += 0.40
            elif cat_name == "Basket":
                if (15 <= cues.get("mean_hue", 0) <= 38) and cues["laplacian_var"] > 350:
                    score += 0.35
            elif cat_name == "Textile":
                if cues["laplacian_var"] > 250 and (cues["aspect_ratio"] > 1.3 or cues["aspect_ratio"] < 0.7):
                    score += 0.25
            elif cat_name == "Jewelry":
                # Jewelry typically has delicate thin components and smaller area
                if (bw * bh) < (0.28 * w * h) or (cues["laplacian_var"] > 450):
                    score += 0.30

            category_scores[cat_name] = round(float(np.clip(score, 0.15, 0.96)), 2)

        # 5. Select highest scoring category
        best_cat = max(category_scores, key=category_scores.get)
        raw_confidence = category_scores[best_cat]

        # 6. Specific craft name selection & Confidence Calibration
        cat_info = HANDICRAFT_TAXONOMY[best_cat]

        if raw_confidence >= 0.85:
            # High confidence: Return specific craft
            if best_cat == "Pottery":
                craft_name = "Terracotta Pottery" if cues["is_terracotta"] else "Ceramic Craft"
            elif best_cat == "Textile":
                craft_name = "Phulkari Dupatta" if cues["laplacian_var"] > 400 else "Handloom Textile"
            elif best_cat == "Jewelry":
                craft_name = "Handcrafted Earrings" if cues["aspect_ratio"] < 0.9 else "Handmade Jewelry"
            elif best_cat == "Metalcraft":
                craft_name = "Brass Craft" if cues["is_brass"] else "Metal Craft"
            elif best_cat == "Basket":
                craft_name = "Handmade Woven Basket"
            elif best_cat == "Woodcraft":
                craft_name = "Wood Carving"
            elif best_cat == "Leather & Bag":
                craft_name = "Handmade Bag"
            else:
                craft_name = cat_info["specific_crafts"][0]

            confidence_label = "detected"
            display_text = f"{craft_name} detected"

        elif raw_confidence >= 0.60:
            # Medium confidence: Return generic category name without hallucinating
            craft_name = cat_info["generic_name"]
            confidence_label = "possible"
            display_text = f"Possible {craft_name.lower()} detected"

        else:
            # Low confidence: Do not guess blindly
            craft_name = "Handicraft item"
            best_cat = "Other handicraft"
            raw_confidence = round(float(np.clip(raw_confidence, 0.45, 0.59)), 2)
            confidence_label = "uncertain"
            display_text = "Product detected, but exact type is uncertain"

        return {
            "detected": True,
            "name": craft_name,
            "category": best_cat,
            "confidence": raw_confidence,
            "confidence_label": confidence_label,
            "display_text": display_text,
            "category_scores": category_scores,
        }


# Global singleton detector
_default_detector: Optional[ProductDetector] = None


def get_detector() -> ProductDetector:
    """Returns singleton detector instance."""
    global _default_detector
    if _default_detector is None:
        _default_detector = ProductDetector()
    return _default_detector


def detect_product(
    rgb_image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    bounding_box: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Convenience functional helper for product detection & classification."""
    detector = get_detector()
    return detector.detect_and_classify(rgb_image, mask=mask, bounding_box=bounding_box)
