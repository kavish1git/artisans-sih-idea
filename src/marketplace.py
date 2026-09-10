"""Marketplace Backdrop Generation and Conservative Enhancement (Modules 8 & 9)
Generates clean, compliant e-commerce backgrounds:
- Pure White (#FFFFFF)
- Off-White (#F8F9FA)
- Light-Gray (#F0F0F2)
- Transparent (RGBA)
Provides conservative, non-destructive photo enhancement:
- Natural sharpness (Unsharp mask)
- Micro-contrast stabilization
- Subtle saturation preservation
"""

from typing import Dict, Any, Union, Optional
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


BACKGROUND_PALETTES = {
    "white": (255, 255, 255),
    "off-white": (248, 249, 250),
    "light-gray": (240, 240, 242),
    "transparent": None,
}


class MarketplaceGenerator:
    """Combines product cutouts onto neutral e-commerce backgrounds with gentle enhancement."""

    @staticmethod
    def enhance_product(
        image: Image.Image,
        sharpness: str = "medium",       # "none", "low", "medium", "high"
        contrast: str = "auto",          # "auto", "low", "medium"
        saturation: str = "low",         # "none", "low", "medium"
        denoise: bool = True,
    ) -> Image.Image:
        """
        Conservative, photo-realistic enhancement. Avoids garish AI artifacts.
        """
        has_alpha = image.mode == "RGBA"
        alpha = image.split()[-1] if has_alpha else None
        rgb_img = image.convert("RGB")

        # 1. Subtle noise reduction on RGB
        if denoise:
            np_rgb = np.array(rgb_img, dtype=np.uint8)
            # Edge-preserving bilateral filter
            np_denoised = cv2.bilateralFilter(np_rgb, d=5, sigmaColor=25, sigmaSpace=25)
            rgb_img = Image.fromarray(np_denoised)

        # 2. Conservative sharpness (Unsharp mask)
        sharpness_factors = {
            "none": 1.0,
            "low": 1.15,
            "medium": 1.30,
            "high": 1.50,
        }
        sharp_mult = sharpness_factors.get(sharpness.lower(), 1.25)
        if sharp_mult != 1.0:
            enhancer = ImageEnhance.Sharpness(rgb_img)
            rgb_img = enhancer.enhance(sharp_mult)

        # 3. Micro-contrast
        contrast_factors = {
            "auto": 1.08,
            "low": 1.05,
            "medium": 1.12,
        }
        cont_mult = contrast_factors.get(contrast.lower(), 1.08)
        if cont_mult != 1.0:
            enhancer = ImageEnhance.Contrast(rgb_img)
            rgb_img = enhancer.enhance(cont_mult)

        # 4. Saturation
        sat_factors = {
            "none": 1.0,
            "low": 1.05,
            "medium": 1.12,
        }
        sat_mult = sat_factors.get(saturation.lower(), 1.05)
        if sat_mult != 1.0:
            enhancer = ImageEnhance.Color(rgb_img)
            rgb_img = enhancer.enhance(sat_mult)

        if has_alpha and alpha is not None:
            rgb_img.putalpha(alpha)

        return rgb_img

    @classmethod
    def apply_background(
        cls,
        rgba_image: Image.Image,
        background_type: str = "white",
    ) -> Image.Image:
        """
        Composites an RGBA product cutout onto a chosen neutral backdrop.
        
        Args:
            rgba_image: RGBA PIL Image.
            background_type: 'white', 'off-white', 'light-gray', or 'transparent'.
            
        Returns:
            RGB Image (or RGBA if 'transparent' requested).
        """
        if background_type.lower() == "transparent":
            return rgba_image.copy()

        bg_color = BACKGROUND_PALETTES.get(background_type.lower(), (255, 255, 255))
        w, h = rgba_image.size

        # Create solid color canvas
        solid_bg = Image.new("RGB", (w, h), bg_color)

        if rgba_image.mode == "RGBA":
            # Paste using alpha mask
            solid_bg.paste(rgba_image, (0, 0), rgba_image)
        else:
            solid_bg.paste(rgba_image, (0, 0))

        return solid_bg


def generate_marketplace_image(
    image: Image.Image,
    background: str = "white",
    sharpness: str = "medium",
    contrast: str = "auto",
    saturation: str = "low",
) -> Image.Image:
    """Convenience helper to enhance and composite onto marketplace backdrop."""
    enhanced = MarketplaceGenerator.enhance_product(
        image, sharpness=sharpness, contrast=contrast, saturation=saturation
    )
    return MarketplaceGenerator.apply_background(enhanced, background_type=background)
