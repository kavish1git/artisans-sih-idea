"""Lighting and Illumination Correction Module (Module 5)
Provides natural, non-destructive lighting correction:
- Contrast Limited Adaptive Histogram Equalization (CLAHE) on luminance channel
- Adaptive Gamma correction for underexposed shadows
- Highlight & shadow tone balancing without color distortion or oversaturation.
"""

from typing import Union, Optional
import cv2
import numpy as np
from PIL import Image


class LightingCorrector:
    """Enhances lighting realistically while preventing artificial AI-style oversaturation."""

    def __init__(self, target_luminance: float = 145.0, max_clahe_clip: float = 2.0):
        self.target_luminance = target_luminance
        self.max_clahe_clip = max_clahe_clip

    def correct_lighting(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Optional[np.ndarray] = None,
        intensity: float = 1.0,
    ) -> Image.Image:
        """
        Corrects uneven lighting, lifts deep shadows, and balances highlights.
        If mask is provided, calculations prioritize foreground product tones.
        
        Args:
            image: RGB PIL Image or BGR/RGBA numpy array.
            mask: Optional 2D binary mask for foreground.
            intensity: Strength of correction (0.0 to 1.5, default 1.0).
            
        Returns:
            Lighting-corrected PIL Image (preserving original mode RGB or RGBA).
        """
        if isinstance(image, Image.Image):
            pil_img = image
            has_alpha = pil_img.mode == "RGBA"
            alpha_layer = pil_img.split()[-1] if has_alpha else None
            rgb_img = pil_img.convert("RGB")
            bgr = cv2.cvtColor(np.array(rgb_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        else:
            has_alpha = image.shape[2] == 4
            alpha_layer = Image.fromarray(image[:, :, 3]) if has_alpha else None
            bgr = image[:, :, :3].copy()

        # Convert to LAB color space to operate strictly on Luminance (L), preventing hue/saturation shifts
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 1. Compute foreground luminance metrics
        if mask is not None and np.sum(mask > 50) > 100:
            fg_pixels = l[mask > 50]
        else:
            fg_pixels = l

        mean_lum = float(np.mean(fg_pixels))

        # 2. Adaptive Gamma Correction: If underexposed, lift shadows smoothly
        if mean_lum < self.target_luminance:
            # gamma < 1.0 brightens midtones without blowing out specular highlights
            gamma = float(np.clip(1.0 - ((self.target_luminance - mean_lum) / 255.0) * 0.8 * intensity, 0.55, 1.0))
            table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
            l = cv2.LUT(l, table)

        # 3. Local illumination leveling via CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clip_limit = max(1.0, min(self.max_clahe_clip, 1.2 + (self.target_luminance - min(mean_lum, self.target_luminance)) / 100.0))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        # Blend CLAHE result gently (65% CLAHE + 35% smooth gamma) to avoid harsh grain
        l_balanced = cv2.addWeighted(l_clahe, 0.65, l, 0.35, 0)

        # Recombine LAB channels and convert back to BGR
        corrected_lab = cv2.merge([l_balanced, a, b])
        corrected_bgr = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        corrected_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)

        result_pil = Image.fromarray(corrected_rgb, mode="RGB")
        if has_alpha and alpha_layer is not None:
            result_pil.putalpha(alpha_layer)

        return result_pil


def correct_lighting(
    image: Union[Image.Image, np.ndarray],
    mask: Optional[np.ndarray] = None,
    intensity: float = 1.0,
) -> Image.Image:
    """Functional convenience wrapper for lighting correction."""
    corrector = LightingCorrector()
    return corrector.correct_lighting(image, mask, intensity)
