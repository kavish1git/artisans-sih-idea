"""Automatic Product Visual Attributes Extractor Module (Module 12)
Extracts objective visual attributes from actual image pixels:
- Dominant color names and hex palette
- Geometric shape via contour morphology
- Spatial orientation
- Surface pattern density
- Surface texture via GLCM co-occurrence analysis
Does NOT hallucinate unverified regional, historical, or material claims.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import cv2
import numpy as np
from PIL import Image

try:
    from skimage.feature import graycomatrix, graycoprops
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


# Reference RGB values for human-readable color naming
NAMED_COLORS = {
    "white": (245, 245, 245),
    "cream": (245, 240, 220),
    "beige": (225, 210, 185),
    "black": (25, 25, 25),
    "gray": (128, 128, 128),
    "brown": (120, 70, 35),
    "terracotta": (180, 85, 45),
    "maroon": (128, 20, 20),
    "red": (210, 40, 40),
    "orange": (235, 120, 30),
    "gold": (215, 175, 45),
    "yellow": (240, 215, 50),
    "mustard": (195, 150, 30),
    "green": (40, 140, 60),
    "olive": (110, 125, 45),
    "emerald": (30, 160, 110),
    "blue": (40, 90, 200),
    "indigo": (45, 50, 130),
    "navy": (20, 35, 80),
    "pink": (230, 120, 150),
    "purple": (120, 50, 150),
}


def _rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
    """Finds closest color name in RGB Euclidean distance with perceptual weights."""
    r, g, b = rgb
    best_name = "gray"
    min_dist = float("inf")

    for name, ref_rgb in NAMED_COLORS.items():
        rr, gg, bb = ref_rgb
        # Weighted Euclidean distance (closer to human eye sensitivity: 2R + 4G + 3B)
        dist = 2 * (r - rr) ** 2 + 4 * (g - gg) ** 2 + 3 * (b - bb) ** 2
        if dist < min_dist:
            min_dist = dist
            best_name = name

    return best_name


class VisualAttributesExtractor:
    """Extracts verifiable visual attributes from product cutout."""

    @staticmethod
    def extract_dominant_colors(
        rgb_image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        k: int = 3,
    ) -> Tuple[List[str], List[str]]:
        """
        Extracts dominant color names and hex codes from foreground pixels.
        Returns: (["brown", "cream"], ["#784623", "#f5f0dc"])
        """
        if mask is not None:
            fg_pixels = rgb_image[mask > 80]
        else:
            fg_pixels = rgb_image.reshape(-1, 3)

        if len(fg_pixels) < 30:
            return ["neutral"], ["#808080"]

        # Subsample for speed
        if len(fg_pixels) > 4000:
            indices = np.random.choice(len(fg_pixels), 4000, replace=False)
            samples = fg_pixels[indices].astype(np.float32)
        else:
            samples = fg_pixels.astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

        counts = np.bincount(labels.flatten())
        sorted_indices = np.argsort(-counts)

        color_names: List[str] = []
        hex_codes: List[str] = []

        for idx in sorted_indices:
            c = centers[idx].astype(int)
            rgb_tuple = (int(c[0]), int(c[1]), int(c[2]))
            hex_str = f"#{rgb_tuple[0]:02x}{rgb_tuple[1]:02x}{rgb_tuple[2]:02x}"
            name = _rgb_to_color_name(rgb_tuple)

            if name not in color_names:
                color_names.append(name)
            hex_codes.append(hex_str)

        return color_names[:k], hex_codes[:k]

    @staticmethod
    def determine_shape(mask: np.ndarray) -> str:
        """Determines geometric 2D shape using contour morphology."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "irregular"

        # Find largest contour
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        if area < 50:
            return "irregular"

        perimeter = float(cv2.arcLength(cnt, True))
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-5)

        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / (hull_area + 1e-5)

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w / max(1, h))
        extent = area / float(w * h + 1e-5)

        # Morphology decision tree
        if circularity > 0.70 and solidity > 0.85 and 0.80 <= aspect_ratio <= 1.25:
            return "round"
        elif 0.82 <= aspect_ratio <= 1.22 and extent > 0.75:
            return "square"
        elif extent > 0.65 and (aspect_ratio > 1.25 or aspect_ratio < 0.80):
            return "rectangular"
        elif circularity > 0.45 and (1.20 <= aspect_ratio <= 2.20 or 0.45 <= aspect_ratio <= 0.83) and solidity > 0.80:
            return "oval"
        elif aspect_ratio < 0.40 or aspect_ratio > 2.50:
            return "elongated"
        elif solidity < 0.70:
            return "irregular"

        return "rectangular" if aspect_ratio > 1.1 or aspect_ratio < 0.9 else "round"

    @staticmethod
    def determine_orientation(bounding_box: Dict[str, int]) -> str:
        """Determines object orientation: vertical, horizontal, or balanced."""
        bw = bounding_box.get("width", 1)
        bh = bounding_box.get("height", 1)
        ratio = bh / max(1, bw)

        if ratio >= 1.18:
            return "upright"
        elif ratio <= 0.85:
            return "horizontal"
        else:
            return "square"

    @staticmethod
    def determine_pattern(rgb_image: np.ndarray, mask: Optional[np.ndarray] = None) -> str:
        """Determines if product has a patterned, plain, or textured surface."""
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        fg_mask = (mask > 80) if mask is not None else np.ones(gray.shape, dtype=bool)

        if np.sum(fg_mask) < 100:
            return "plain"

        # Edge gradient density
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)

        mean_grad = float(np.mean(grad_mag[fg_mask]))
        std_grad = float(np.std(grad_mag[fg_mask]))

        # Color variance across product area
        fg_pixels = rgb_image[fg_mask]
        color_std = float(np.mean(np.std(fg_pixels, axis=0)))

        if mean_grad > 28.0 or (mean_grad > 18.0 and color_std > 38.0):
            return "patterned"
        elif mean_grad < 12.0 and color_std < 22.0:
            return "plain"
        else:
            return "subtly textured"

    @staticmethod
    def determine_texture(rgb_image: np.ndarray, mask: Optional[np.ndarray] = None) -> str:
        """Determines surface texture via GLCM co-occurrence analysis."""
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        fg_mask = (mask > 80) if mask is not None else np.ones(gray.shape, dtype=bool)

        if np.sum(fg_mask) < 100:
            return "smooth"

        # Check for metallic specular highlights
        v_channel = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)[:, :, 2]
        high_specular_ratio = float(np.sum((v_channel > 240) & fg_mask) / np.sum(fg_mask))
        if high_specular_ratio > 0.08:
            return "metallic / polished"

        if not _HAS_SKIMAGE:
            # Fallback texture estimation using Laplacian variance
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if lap_var > 400:
                return "woven-looking"
            elif lap_var < 100:
                return "smooth"
            else:
                return "textured"

        try:
            # Downscale small patch for ultra-fast GLCM
            patch = cv2.resize(gray, (128, 128))
            glcm = graycomatrix(
                patch,
                distances=[1, 3],
                angles=[0, np.pi / 4, np.pi / 2],
                levels=256,
                symmetric=True,
                normed=True,
            )
            contrast = float(graycoprops(glcm, "contrast").mean())
            homogeneity = float(graycoprops(glcm, "homogeneity").mean())
            energy = float(graycoprops(glcm, "energy").mean())

            if contrast > 180 and homogeneity < 0.45:
                return "woven-looking"
            elif homogeneity > 0.65:
                return "smooth / glazed"
            elif contrast > 100:
                return "carved / granular"
            else:
                return "textured"
        except Exception:
            return "textured"

    @classmethod
    def extract_all(
        cls,
        rgb_image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        bounding_box: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """Runs complete visual attribute extraction pipeline."""
        colors, hex_palette = cls.extract_dominant_colors(rgb_image, mask=mask, k=3)

        shape = cls.determine_shape(mask) if mask is not None else "rectangular"
        orientation = cls.determine_orientation(bounding_box) if bounding_box else "upright"
        pattern = cls.determine_pattern(rgb_image, mask=mask)
        texture = cls.determine_texture(rgb_image, mask=mask)

        return {
            "dominant_colors": colors,
            "palette_hex": hex_palette,
            "shape": shape,
            "orientation": orientation,
            "pattern": pattern,
            "texture": texture,
        }


def extract_attributes(
    rgb_image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    bounding_box: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Convenience helper for visual attribute extraction."""
    return VisualAttributesExtractor.extract_all(rgb_image, mask=mask, bounding_box=bounding_box)
