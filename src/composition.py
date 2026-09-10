"""Composition Analysis Module
Determines whether an uploaded photograph should be preserved as a full composition
(e.g., flatlay, textile stack, multi-item set, clean studio shot, carpet, wall hanging)
or whether foreground background removal should be applied.
Prevents destructive under-segmentation and ruined photos.
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


def analyze_composition_suitability(
    bgr_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    category: Optional[str] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Evaluates whether an image is a full craft composition that should be preserved as a whole,
    or whether foreground background removal is appropriate.

    Returns:
        (preserve_full_image: bool, reason: str, metrics: Dict[str, Any])
    """
    h, w = bgr_img.shape[:2]
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    total_pixels = float(h * w)
    total_edge_density = float(np.mean(edges > 0))
    total_color_std = float(np.mean(np.std(bgr_img, axis=(0, 1))))

    # Corner analysis to detect clean studio backdrops
    c_h = max(5, int(h * 0.12))
    c_w = max(5, int(w * 0.20))
    corners = np.concatenate([
        bgr_img[:c_h, :c_w].reshape(-1, 3),
        bgr_img[:c_h, -c_w:].reshape(-1, 3),
    ])
    corner_mean = float(np.mean(corners))
    corner_std = float(np.mean(np.std(corners, axis=0)))

    # Border analysis
    border_pixels = np.concatenate([
        bgr_img[0, :], bgr_img[-1, :], bgr_img[:, 0], bgr_img[:, -1]
    ])
    border_std = float(np.mean(np.std(border_pixels, axis=0)))

    # Mask evaluation (if segmentation mask exists)
    fg_pct = 100.0
    bg_edge_density = 0.0
    bg_color_std = 0.0
    halo_texture_var = 0.0

    if mask is not None:
        bin_m = (mask > 40).astype(np.uint8)
        fg_count = int(np.sum(bin_m))
        fg_pct = round(float((fg_count / total_pixels) * 100.0), 1)

        bg_mask = (bin_m == 0)
        if np.sum(bg_mask) > 100:
            bg_edge_density = float(np.mean(edges[bg_mask] > 0))
            bg_color_std = float(np.mean(np.std(bgr_img[bg_mask], axis=0)))

            # Outer halo of the mask (region right next to cutout)
            kernel_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            dilated = cv2.dilate(bin_m, kernel_out)
            outer_halo = (dilated == 1) & (bin_m == 0)
            if np.sum(outer_halo) > 50:
                halo_texture_var = float(np.var(gray[outer_halo]))

    metrics = {
        "fg_pct": fg_pct,
        "total_edge_density": round(total_edge_density, 3),
        "total_color_std": round(total_color_std, 1),
        "bg_edge_density": round(bg_edge_density, 3),
        "bg_color_std": round(bg_color_std, 1),
        "halo_texture_var": round(halo_texture_var, 1),
        "corner_mean": round(corner_mean, 1),
        "corner_std": round(corner_std, 1),
        "border_std": round(border_std, 1),
    }

    # -------------------------------------------------------------
    # Rule 1: Catastrophic Under-Segmentation Guard
    # If the mask captures < 65% of the frame, but the discarded background
    # region has high edge density and high color standard deviation,
    # the model is slicing through a multi-layer craft (e.g. folded textile stack).
    # -------------------------------------------------------------
    if mask is not None and fg_pct < 65.0:
        if (bg_edge_density > 0.08 and bg_color_std > 30.0) or halo_texture_var > 3500.0:
            return (
                True,
                "The image contains a multi-part craft composition or textile stack. Preserving the complete photograph to avoid cutting off parts of the product.",
                metrics,
            )

    # -------------------------------------------------------------
    # Rule 2: Full-Frame Craft / Flatlay / Textile Stack
    # When edges and colors fill the entire frame, or borders touch rich craft content
    # -------------------------------------------------------------
    is_textile_or_decor = category and any(
        c in category.lower() for c in ["textile", "saree", "dupatta", "shawl", "carpet", "painting", "art"]
    )

    if is_textile_or_decor and total_edge_density > 0.15 and total_color_std > 45.0:
        return (
            True,
            "Full-frame textile or artisan scene detected. Preserving complete product arrangement.",
            metrics,
        )

    # -------------------------------------------------------------
    # Rule 3: High Coverage Studio Shot
    # The product fills > 80% of the image, so cutout is unnecessary and risky
    # -------------------------------------------------------------
    if fg_pct > 80.0 and corner_mean > 160.0:
        return (
            True,
            "Product already occupies the full frame on a clean studio setting.",
            metrics,
        )

    return (
        False,
        "Isolated product on generic surface. Background removal applied.",
        metrics,
    )
