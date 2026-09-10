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

    @classmethod

    def select_automatic_background(cls, rgba_image: Image.Image) -> str:
        """
        Intelligently decides background color based on product luminance:
        - Very light / white crafts (e.g. white marble, ivory, cream lace): 'light-gray' (#F0F0F2) to prevent edge bleed.
        - Darker or vibrant crafts: clean 'off-white' (#F8F9FA).
        """
        if rgba_image.mode != "RGBA":
            return "off-white"

        arr = np.array(rgba_image)
        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3]

        fg_mask = alpha > 80
        if np.sum(fg_mask) < 50:
            return "off-white"

        # Calculate luminance in CIE-LAB space
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]  # L in [0, 255] in OpenCV, maps to [0, 100]

        fg_lightness = float(np.mean(l_channel[fg_mask])) * (100.0 / 255.0)

        # Inspect edge pixels to prevent boundary washout
        eroded_mask = cv2.erode((fg_mask).astype(np.uint8), np.ones((5, 5), np.uint8))
        edge_mask = (fg_mask.astype(np.uint8) - eroded_mask) > 0
        if np.sum(edge_mask) > 20:
            edge_lightness = float(np.mean(l_channel[edge_mask])) * (100.0 / 255.0)
        else:
            edge_lightness = fg_lightness

        # If product or product edges are very bright/white (> 80), use light-gray for contrast
        if fg_lightness > 80.0 or edge_lightness > 83.0:
            return "light-gray"
        else:
            return "off-white"


def generate_marketplace_image(
    image: Image.Image,
    background: Optional[str] = None,
    sharpness: str = "medium",
    contrast: str = "auto",
    saturation: str = "low",
) -> Image.Image:
    """Convenience helper to enhance and composite onto marketplace backdrop."""
    enhanced = MarketplaceGenerator.enhance_product(
        image, sharpness=sharpness, contrast=contrast, saturation=saturation
    )
    if background is None or background == "auto":
        bg_type = MarketplaceGenerator.select_automatic_background(image)
    else:
        bg_type = background
    return MarketplaceGenerator.apply_background(enhanced, background_type=bg_type)

