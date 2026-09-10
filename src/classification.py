"""Product Classification Module (Stage 2)
Classifies cropped artisan products into Indian handicraft categories
using deep vision representations, temperature-scaled confidence calibration,
and safe unknown/fallback handling.
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


# Standardized Artisan Taxonomy Categories (Requirement 2)
ARTISAN_CATEGORIES = [
    "pottery",
    "terracotta_pottery",
    "phulkari_dupatta",
    "saree",
    "handloom_textile",
    "shawl",
    "embroidery",
    "basket",
    "bamboo_craft",
    "wooden_craft",
    "wood_carving",
    "metal_craft",
    "brass_craft",
    "jewelry",
    "handmade_bag",
    "leather_craft",
    "painting",
    "sculpture",
    "traditional_toy",
    "home_decor",
    "other_handicraft",
    "unknown",
]

# Accurate ImageNet synset mappings for artisan categories
CATEGORY_SYNSET_MAPPING = {
    "pottery": [883, 605, 899, 898, 710, 849, 968, 659, 504],   # 883=vase, 605=pitcher, 899=water jug, 898=water bottle
    "terracotta_pottery": [883, 605, 899, 898, 710],              # Earthy clay vessels
    "phulkari_dupatta": [834, 783, 815, 614],                     # Stole, poncho, lace/embroidery, kimono
    "saree": [804, 614, 834],                                     # Sarong, drape, stole
    "handloom_textile": [915, 834, 783, 407],                     # Wool, stole, textile drape
    "shawl": [834, 783, 915],                                     # Stole, poncho, woolen fabric
    "embroidery": [815, 834, 783],                                # Intricate threadwork/lace
    "basket": [431, 790, 563],                                    # 431=hamper, 790=shopping basket
    "bamboo_craft": [431, 790, 875],                              # Woven cane, bamboo pole
    "wooden_craft": [919, 514, 403, 770],                         # Wooden spoon, cradle, acoustic guitar
    "wood_carving": [875, 919, 514],                              # Totem pole, carved wood
    "metal_craft": [427, 468, 572, 498, 594, 466, 444],          # Bell, cauldron, goblet, cleaver, hook
    "brass_craft": [427, 468, 572, 594],                         # Brass bell, goblet, urn
    "jewelry": [865, 509, 760, 487],                             # Trinket/jewelry, hair slide, safety pin/buckle
    "handmade_bag": [748, 637, 414],                             # Purse, mailbag, backpack
    "leather_craft": [748, 803, 477],                             # Leather pouch, sandal, boot
    "painting": [917, 922],                                       # Comic book / graphic art, print
    "sculpture": [875, 643],                                      # Totem pole, ceremonial mask
    "traditional_toy": [840, 722],                                # Teddy, puppet, toy
    "home_decor": [470, 682, 721],                                # Candle/candlestick, decorative pillow
}


MAJOR_CATEGORIES = {
    "pottery": "Pottery",
    "terracotta_pottery": "Pottery",
    "phulkari_dupatta": "Textile",
    "saree": "Textile",
    "handloom_textile": "Textile",
    "shawl": "Textile",
    "embroidery": "Textile",
    "basket": "Basket",
    "bamboo_craft": "Wood & Bamboo",
    "wooden_craft": "Wood & Bamboo",
    "wood_carving": "Wood & Bamboo",
    "metal_craft": "Metal & Brass",
    "brass_craft": "Metal & Brass",
    "jewelry": "Jewelry",
    "handmade_bag": "Leather & Bag",
    "leather_craft": "Leather & Bag",
    "painting": "Art & Decor",
    "sculpture": "Art & Decor",
    "traditional_toy": "Art & Decor",
    "home_decor": "Art & Decor",
    "other_handicraft": "Other Handicraft",
    "unknown": "Unknown",
}


class ProductClassifier:
    """Two-stage classifier with temperature scaling and calibrated confidence."""

    def __init__(self, model_path: Optional[str] = None, temperature: float = 2.5):
        self.temperature = temperature
        self.session = None
        default_model = Path(__file__).parent.parent / "models" / "mobilenetv2-7.onnx"

        if model_path and os.path.exists(model_path):
            target = Path(model_path)
        else:
            target = default_model

        if _HAS_ORT and target.exists():
            try:
                self.session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
            except Exception:
                self.session = None

    def _preprocess(self, cropped_rgb: np.ndarray) -> np.ndarray:
        """Preprocesses cropped product for classification."""
        resized = cv2.resize(cropped_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        return np.transpose(arr, (2, 0, 1))[np.newaxis, ...]

    def classify_crop(
        self,
        cropped_rgb: np.ndarray,
        cropped_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Classifies the cropped product region.
        Returns:
            category: str (high-level artisan group: Pottery, Textile, etc.)
            subcategory: str (fine-grained craft taxonomy)
            craft_type: str (synonym for subcategory)
            confidence: float (calibrated 0.0 - 1.0)
            confidence_status: 'high', 'medium', 'low'
            category_logits: dict of raw class scores
        """
        if cropped_rgb.size == 0 or cropped_rgb.shape[0] < 10 or cropped_rgb.shape[1] < 10:
            return {
                "category": "Unknown",
                "subcategory": "unknown",
                "craft_type": "unknown",
                "confidence": 0.0,
                "confidence_status": "low",
                "object_name": "Unknown",
                "explanation": "No clear product detected in crop.",
            }

        # 1. Compute deep vision logits
        if self.session is None:
            return {
                "category": "Other Handicraft",
                "subcategory": "other_handicraft",
                "craft_type": "other_handicraft",
                "confidence": 0.50,
                "confidence_status": "low",
                "object_name": "Handicraft Product",
            }

        input_tensor = self._preprocess(cropped_rgb)
        ort_inputs = {self.session.get_inputs()[0].name: input_tensor}
        raw_logits = self.session.run(None, ort_inputs)[0][0]

        # 2. Aggregate logits per artisan category
        category_raw_scores: Dict[str, float] = {}
        for cat, synsets in CATEGORY_SYNSET_MAPPING.items():
            synset_logits = [raw_logits[s] for s in synsets if s < len(raw_logits)]
            category_raw_scores[cat] = float(max(synset_logits)) if synset_logits else -10.0

        # 3. Incorporate visual domain cues on foreground
        hsv = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2HSV)
        fg_mask = (cropped_mask > 50) if cropped_mask is not None else np.ones(cropped_rgb.shape[:2], dtype=bool)

        if np.sum(fg_mask) > 30:
            fg_hsv = hsv[fg_mask]
            mean_hue = float(np.median(fg_hsv[:, 0]))
            mean_sat = float(np.mean(fg_hsv[:, 1]))
            mean_val = float(np.mean(fg_hsv[:, 2]))

            # Terracotta / earthenware boost (H: 8-24, S: > 45)
            if (8 <= mean_hue <= 24) and mean_sat > 45:
                category_raw_scores["terracotta_pottery"] += 1.5
                category_raw_scores["pottery"] += 1.0

            # Brass / metallic boost
            if (22 <= mean_hue <= 42) and mean_sat > 40 and mean_val > 90:
                category_raw_scores["brass_craft"] += 1.5
                category_raw_scores["metal_craft"] += 1.0

        # 4. Temperature-Scaled Softmax over artisan categories
        categories = list(category_raw_scores.keys())
        logits_vector = np.array([category_raw_scores[c] for c in categories], dtype=np.float32)

        # Apply temperature scaling to prevent overconfidence
        scaled_logits = logits_vector / max(0.1, self.temperature)
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probabilities = exp_logits / np.sum(exp_logits)

        # 5. Top prediction and major-group probability
        top_indices = np.argsort(probabilities)[::-1]
        best_idx = top_indices[0]
        best_category = categories[best_idx]
        best_prob = float(probabilities[best_idx])
        major_category = MAJOR_CATEGORIES.get(best_category, "Other Handicraft")

        # Sum probabilities for the winning major category (e.g. terracotta_pottery + pottery)
        major_prob = float(sum(
            probabilities[i] for i, c in enumerate(categories)
            if MAJOR_CATEGORIES.get(c) == major_category
        ))

        # Find best probability from an alternative major category
        alt_major_prob = 0.0
        for i in top_indices:
            if MAJOR_CATEGORIES.get(categories[i]) != major_category:
                alt_major_prob = float(probabilities[i])
                break

        major_margin = major_prob - alt_major_prob

        # 6. Confidence Calibration & Thresholding (Requirement 3)
        # Never accept a high confidence unless the margin over alternative craft categories is significant
        if major_prob >= 0.45 and major_margin >= 0.15:
            confidence_status = "high"
            calibrated_conf = round(float(np.clip(0.85 + (major_margin * 0.2), 0.85, 0.95)), 2)
        elif major_prob >= 0.25 and major_margin >= 0.05:
            confidence_status = "medium"
            calibrated_conf = round(float(np.clip(0.65 + (major_margin * 0.3), 0.60, 0.84)), 2)
        else:
            confidence_status = "low"
            calibrated_conf = round(float(np.clip(major_prob, 0.35, 0.59)), 2)
            # If low confidence, do NOT force exact sub-class
            if "pottery" in best_category:
                best_category = "pottery"
            elif "dupatta" in best_category or "saree" in best_category:
                best_category = "handloom_textile"
            else:
                best_category = "other_handicraft"

        # Human-friendly display name
        display_names = {
            "pottery": "Pottery Craft",
            "terracotta_pottery": "Terracotta Pottery",
            "phulkari_dupatta": "Phulkari Dupatta",
            "saree": "Traditional Saree",
            "handloom_textile": "Handloom Textile",
            "shawl": "Artisan Shawl",
            "embroidery": "Embroidered Craft",
            "basket": "Handmade Basket",
            "bamboo_craft": "Bamboo Craft",
            "wooden_craft": "Wooden Craft",
            "wood_carving": "Wood Carving",
            "metal_craft": "Metal Craft",
            "brass_craft": "Brass Craft",
            "jewelry": "Handmade Jewelry",
            "handmade_bag": "Handmade Bag",
            "leather_craft": "Leather Craft",
            "painting": "Traditional Painting",
            "sculpture": "Sculpture",
            "traditional_toy": "Traditional Toy",
            "home_decor": "Home Decor",
            "other_handicraft": "Handicraft Item",
            "unknown": "Unknown Craft",
        }

        object_name = display_names.get(best_category, "Handicraft Item")
        major_category = MAJOR_CATEGORIES.get(best_category, "Other Handicraft")

        return {
            "category": major_category,
            "subcategory": best_category,
            "craft_type": best_category,
            "object_name": object_name,
            "confidence": calibrated_conf,
            "confidence_status": confidence_status,
            "top_categories": [
                {"category": categories[i], "probability": round(float(probabilities[i]), 3)}
                for i in top_indices[:4]
            ],
        }


# Singleton classifier
_default_classifier: Optional[ProductClassifier] = None


def get_classifier() -> ProductClassifier:
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = ProductClassifier()
    return _default_classifier


def classify_product(
    cropped_rgb: np.ndarray,
    cropped_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Convenience functional helper for product classification."""
    classifier = get_classifier()
    return classifier.classify_crop(cropped_rgb, cropped_mask=cropped_mask)
