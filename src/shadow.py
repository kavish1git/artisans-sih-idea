"""Shadow Handling and Grounding Module (Module 4)
Supports:
- 'professional': Clean background with subtle studio grounding contact shadow
- 'natural': Softer ambient occlusion shadow preserving more environmental realism
Prevents artificial "floating sticker" look while maintaining e-commerce standards.
"""

from typing import Tuple, Dict, Any, Union, Optional
import cv2
import numpy as np
from PIL import Image, ImageFilter


class ShadowHandler:
    """Synthesizes natural grounding shadows for isolated product cutouts."""

    def __init__(self, mode: str = "professional"):
        self.mode = mode

    def add_grounding_shadow(
        self,
        transparent_product: Image.Image,
        bbox: Optional[Dict[str, int]] = None,
        shadow_intensity: float = 0.30,
        blur_radius: int = 15,
        vertical_offset: int = 8,
    ) -> Image.Image:
        """
        Adds a natural contact grounding shadow beneath the product on a transparent canvas.
        
        Args:
            transparent_product: RGBA PIL Image.
            bbox: Bounding box {'x': x, 'y': y, 'width': w, 'height': h}.
            shadow_intensity: Shadow opacity (0.0 to 1.0).
            blur_radius: Softness of shadow edge.
            vertical_offset: Downward displacement of shadow.
            
        Returns:
            RGBA PIL Image with product and composite grounding shadow.
        """
        if transparent_product.mode != "RGBA":
            return transparent_product

        w, h = transparent_product.size
        alpha = np.array(transparent_product.split()[-1])

        # 1. Determine base of product
        if bbox:
            bx = bbox.get("x", 0)
            by = bbox.get("y", 0)
            bw = bbox.get("width", w)
            bh = bbox.get("height", h)
        else:
            y_indices, x_indices = np.where(alpha > 20)
            if len(y_indices) == 0:
                return transparent_product
            bx = int(np.min(x_indices))
            by = int(np.min(y_indices))
            bw = int(np.max(x_indices) - bx)
            bh = int(np.max(y_indices) - by)

        # 2. Construct shadow layer
        shadow_canvas = np.zeros((h, w), dtype=np.uint8)

        if self.mode == "professional":
            # Tight, elegant elliptical contact shadow under the product base
            center_x = bx + bw // 2
            center_y = min(h - 5, by + bh + vertical_offset // 2)
            axis_x = max(10, int(bw * 0.42))
            axis_y = max(4, int(bh * 0.05))

            # Primary contact shadow
            cv2.ellipse(shadow_canvas, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 255, -1)
            # Secondary wider soft penumbra
            penumbra = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(penumbra, (center_x, center_y + 2), (int(axis_x * 1.3), int(axis_y * 1.8)), 0, 0, 360, 160, -1)
            shadow_canvas = cv2.add(shadow_canvas, penumbra)

        else:  # 'natural' mode
            # Cast shadow extracted from lower 20% silhouette of the product
            lower_cutoff = by + int(bh * 0.80)
            base_mask = np.zeros_like(alpha)
            base_mask[lower_cutoff:by + bh, bx:bx + bw] = alpha[lower_cutoff:by + bh, bx:bx + bw]

            # Squash vertically and displace downward
            M = np.float32([[1, 0, 0], [0, 0.4, vertical_offset]])
            shadow_canvas = cv2.warpAffine(base_mask, M, (w, h))

        # 3. Apply Gaussian blur to produce soft natural falloff
        ksize = max(3, blur_radius * 2 + 1)
        shadow_blurred = cv2.GaussianBlur(shadow_canvas, (ksize, ksize), blur_radius / 2.0)

        # Scale shadow alpha by requested intensity
        shadow_alpha = (shadow_blurred.astype(np.float32) * shadow_intensity).astype(np.uint8)

        # 4. Create RGBA shadow image (neutral dark umber / charcoal: 30, 30, 30)
        shadow_img = Image.new("RGBA", (w, h), (30, 30, 30, 0))
        shadow_alpha_pil = Image.fromarray(shadow_alpha, mode="L")
        shadow_img.putalpha(shadow_alpha_pil)

        # 5. Composite: Shadow on bottom, product on top
        composite = Image.alpha_composite(shadow_img, transparent_product)
        return composite


def apply_grounding_shadow(
    image: Image.Image,
    bbox: Optional[Dict[str, int]] = None,
    mode: str = "professional",
    intensity: float = 0.30,
) -> Image.Image:
    """Convenience functional wrapper for grounding shadow synthesis."""
    handler = ShadowHandler(mode=mode)
    return handler.add_grounding_shadow(image, bbox=bbox, shadow_intensity=intensity)
