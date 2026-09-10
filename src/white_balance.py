"""White Balance and Color Constancy Module (Module 6)
Adaptive color temperature correction to eliminate:
- Warm / yellow incandescent indoor lighting
- Cool / blue shadows and fluorescent tints
Uses Shades-of-Gray (Minkowski p-norm) color constancy while preserving
authentic artisan dyes and handicraft material colors.
"""

from typing import Union, Optional
import cv2
import numpy as np
from PIL import Image


class WhiteBalanceCorrector:
    """Adaptive color constancy to restore true material hues."""

    def __init__(self, p_norm: int = 6):
        # p=6 approximates the human visual system's Shades-of-Gray adaptation
        self.p_norm = p_norm

    def correct(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Optional[np.ndarray] = None,
        strength: float = 0.85,
    ) -> Image.Image:
        """
        Applies adaptive white balance.
        
        Args:
            image: RGB PIL Image or BGR/RGBA numpy array.
            mask: Optional 2D foreground mask.
            strength: Interpolation strength between original and corrected (0.0 - 1.0).
            
        Returns:
            White-balanced PIL Image.
        """
        if isinstance(image, Image.Image):
            pil_img = image
            has_alpha = pil_img.mode == "RGBA"
            alpha_layer = pil_img.split()[-1] if has_alpha else None
            rgb = np.array(pil_img.convert("RGB"), dtype=np.float32)
        else:
            has_alpha = image.shape[2] == 4
            alpha_layer = Image.fromarray(image[:, :, 3]) if has_alpha else None
            rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)

        # 1. Isolate pixels to estimate illuminant
        if mask is not None and np.sum(mask > 50) > 100:
            active_pixels = rgb[mask > 50]
        else:
            # Exclude extreme highlights and deep blacks from illuminant estimation
            gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            valid_mask = (gray > 15.0) & (gray < 240.0)
            if np.sum(valid_mask) > 100:
                active_pixels = rgb[valid_mask]
            else:
                active_pixels = rgb.reshape(-1, 3)

        # 2. Compute Minkowski p-norm for each channel (Shades-of-Gray)
        p = self.p_norm
        norm_r = float(np.power(np.mean(np.power(active_pixels[:, 0], p)), 1.0 / p))
        norm_g = float(np.power(np.mean(np.power(active_pixels[:, 1], p)), 1.0 / p))
        norm_b = float(np.power(np.mean(np.power(active_pixels[:, 2], p)), 1.0 / p))

        # Target gray is the Euclidean geometric mean across channels
        gray_target = (norm_r + norm_g + norm_b) / 3.0

        # Avoid zero-division
        scale_r = gray_target / max(norm_r, 1e-3)
        scale_g = gray_target / max(norm_g, 1e-3)
        scale_b = gray_target / max(norm_b, 1e-3)

        # Dampen extreme scale factors to preserve vibrant natural handicraft colors
        max_scale_dev = 0.35  # Max 35% adjustment
        scale_r = float(np.clip(scale_r, 1.0 - max_scale_dev, 1.0 + max_scale_dev))
        scale_g = float(np.clip(scale_g, 1.0 - max_scale_dev, 1.0 + max_scale_dev))
        scale_b = float(np.clip(scale_b, 1.0 - max_scale_dev, 1.0 + max_scale_dev))

        # 3. Apply color gains with strength interpolation
        corrected = np.empty_like(rgb)
        corrected[:, :, 0] = rgb[:, :, 0] * (1.0 + (scale_r - 1.0) * strength)
        corrected[:, :, 1] = rgb[:, :, 1] * (1.0 + (scale_g - 1.0) * strength)
        corrected[:, :, 2] = rgb[:, :, 2] * (1.0 + (scale_b - 1.0) * strength)

        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        result_pil = Image.fromarray(corrected, mode="RGB")

        if has_alpha and alpha_layer is not None:
            result_pil.putalpha(alpha_layer)

        return result_pil


def correct_white_balance(
    image: Union[Image.Image, np.ndarray],
    mask: Optional[np.ndarray] = None,
    strength: float = 0.85,
) -> Image.Image:
    """Convenience functional wrapper for white balance correction."""
    corrector = WhiteBalanceCorrector()
    return corrector.correct(image, mask, strength)
