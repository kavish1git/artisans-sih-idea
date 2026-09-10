"""Background Removal Module (Module 3)
Generates high-fidelity transparent PNG with preserved edge details, threads,
and embroidery.
"""

from typing import Union, Optional, Tuple
from pathlib import Path
import io
import cv2
import numpy as np
from PIL import Image

from src.utils import load_image_safely
from src.segmentation import ProductSegmenter, SegmentationResult, SegmentationError


class BackgroundRemover:
    """Produces clean transparent PNG images from raw product photos."""

    def __init__(self, segmenter: Optional[ProductSegmenter] = None):
        self.segmenter = segmenter or ProductSegmenter()

    def remove_background(
        self,
        source: Union[str, Path, bytes, np.ndarray, Image.Image],
        feather_edges: bool = True,
    ) -> Tuple[Image.Image, SegmentationResult]:
        """
        Extracts the foreground product and returns a transparent RGBA PIL Image
        along with segmentation metadata.
        
        Args:
            source: Image input (path, bytes, array, or PIL Image).
            feather_edges: If True, applies subtle sub-pixel anti-aliasing to perimeter.
            
        Returns:
            Tuple of (transparent_pil_image, segmentation_result)
        """
        seg_result = self.segmenter.segment(source)
        mask = seg_result.mask

        # Validate that product exists in mask
        fg_pixel_count = np.sum(mask > 32)
        if fg_pixel_count < 100:
            raise SegmentationError(
                "Background removal failed: No distinct product was detected. Please ensure product contrasts with background."
            )

        rgba = seg_result.rgba_foreground.copy()

        if feather_edges:
            # Gentle edge feathering to eliminate harsh pixel staircasing while preserving threads
            alpha = rgba[:, :, 3].astype(np.float32) / 255.0
            # Subtle Gaussian edge blur for alpha channel only
            alpha_feathered = cv2.GaussianBlur(alpha, (3, 3), 0.5)
            # Only blend transitional edge pixels (0.05 < alpha < 0.95)
            transitional = (alpha > 0.05) & (alpha < 0.95)
            alpha[transitional] = alpha_feathered[transitional]
            rgba[:, :, 3] = (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)

        transparent_pil = Image.fromarray(rgba, mode="RGBA")
        return transparent_pil, seg_result

    def save_transparent_png(
        self,
        source: Union[str, Path, bytes, np.ndarray, Image.Image],
        output_path: Union[str, Path],
    ) -> SegmentationResult:
        """Removes background and saves the result directly to PNG file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img, seg_result = self.remove_background(source)
        img.save(str(output_path), format="PNG", optimize=True)
        return seg_result


def remove_background(
    source: Union[str, Path, bytes, np.ndarray, Image.Image],
) -> Tuple[Image.Image, SegmentationResult]:
    """Convenience wrapper for background removal."""
    remover = BackgroundRemover()
    return remover.remove_background(source)
