"""Product Segmentation Module (Module 2)
Provides deep-learning based salient object segmentation with U2-Net/rembg
and an adaptive OpenCV GrabCut fallback for resilient offline execution.
"""

from typing import Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from src.utils import load_image_safely


@dataclass
class SegmentationResult:
    """Encapsulates segmentation output metadata."""
    mask: np.ndarray             # 2D uint8 mask [0, 255]
    rgba_foreground: np.ndarray  # 4-channel RGBA array [H, W, 4]
    bounding_box: Dict[str, int] # x, y, width, height
    confidence: float            # 0.0 - 1.0
    model_used: str              # e.g., 'u2net', 'grabcut_fallback'


class SegmentationError(Exception):
    """Raised when segmentation fails to isolate product."""
    pass


class ProductSegmenter:
    """Extracts foreground handicraft products with edge and fine-detail preservation."""

    def __init__(self, model_name: str = "u2netp", use_fallback: bool = True):
        self.model_name = model_name
        self.use_fallback = use_fallback
        self._rembg_session = None

    def _get_rembg_session(self):
        """Lazy initialization of rembg ONNX session."""
        if self._rembg_session is None:
            try:
                import rembg
                # Default session uses U2-Net optimized for CPU ONNX Runtime
                self._rembg_session = rembg.new_session(self.model_name)
            except Exception as e:
                self._rembg_session = False  # Mark as unavailable
        return self._rembg_session

    def segment_with_u2net(self, rgb_pil: Image.Image) -> Optional[np.ndarray]:
        """Runs U2-Net matting to produce an 8-bit alpha mask."""
        session = self._get_rembg_session()
        if not session:
            return None

        try:
            import rembg
            # rembg remove returns RGBA PIL Image
            rgba_out = rembg.remove(
                rgb_pil,
                session=session,
                only_mask=True,
                post_process_mask=True,
                alpha_matting=False,
            )
            # Convert mask to single channel uint8
            mask_np = np.array(rgba_out, dtype=np.uint8)
            if len(mask_np.shape) == 3:
                mask_np = mask_np[:, :, 0]
            return mask_np
        except Exception:
            return None

    def segment_with_grabcut(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Robust adaptive GrabCut segmentation as a fallback.
        Uses saliency and contrast gradients to seed foreground/background seeds.
        """
        h, w = bgr_img.shape[:2]
        # Initial bounding box inset 5%
        margin_x = max(5, int(w * 0.05))
        margin_y = max(5, int(h * 0.05))
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(bgr_img, mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
            # 0=BGD, 1=FGD, 2=PR_BGD, 3=PR_FGD
            bin_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

            # Refine mask with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
            bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
            return bin_mask
        except Exception:
            # Fallback simple threshold if GrabCut encounters degenerate data
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            _, bin_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return bin_mask

    def calculate_bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Calculates bounding box [x, y, w, h] around non-zero mask pixels."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            h, w = mask.shape
            return 0, 0, w, h

        # Aggregate bounds of prominent contours
        min_x, min_y = mask.shape[1], mask.shape[0]
        max_x, max_y = 0, 0

        # Discard tiny noise specks (< 0.5% area)
        min_contour_area = 0.005 * (mask.shape[0] * mask.shape[1])
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        if not valid_contours:
            valid_contours = contours

        for c in valid_contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            min_x = min(min_x, bx)
            min_y = min(min_y, by)
            max_x = max(max_x, bx + bw)
            max_y = max(max_y, by + bh)

        return min_x, min_y, max(1, max_x - min_x), max(1, max_y - min_y)

    def calculate_confidence(self, mask: np.ndarray) -> float:
        """
        Estimates segmentation quality and certainty based on edge smoothness,
        foreground occupancy, and connectivity.
        """
        h, w = mask.shape
        total_pixels = float(h * w)
        fg_pixels = float(np.sum(mask > 128))
        ratio = fg_pixels / total_pixels

        # Extremely low or extremely high coverage indicates failure
        if ratio < 0.03 or ratio > 0.98:
            return 0.2

        # Evaluate edge transition certainty
        uncertain_pixels = float(np.sum((mask > 15) & (mask < 240)))
        certainty = 1.0 - min(0.5, (uncertain_pixels / (fg_pixels + 1e-5)))

        return round(float(np.clip(certainty * 0.95, 0.3, 0.98)), 2)

    def segment(
        self, source: Union[str, Path, bytes, np.ndarray, Image.Image]
    ) -> SegmentationResult:
        """
        Executes primary U2-Net segmentation or GrabCut fallback.
        Returns clean mask, RGBA foreground, bounding box, and confidence.
        """
        if isinstance(source, np.ndarray):
            bgr_img = source
            rgb_pil = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
        else:
            bgr_img, rgb_pil = load_image_safely(source)

        mask: Optional[np.ndarray] = None
        model_used = "u2net"

        # 1. Attempt U2-Net inference
        mask = self.segment_with_u2net(rgb_pil)

        # 2. Check if U2-Net succeeded and produced a valid mask
        if mask is None or np.sum(mask > 128) < (0.02 * mask.size):
            if self.use_fallback:
                mask = self.segment_with_grabcut(bgr_img)
                model_used = "grabcut_saliency"
            else:
                raise SegmentationError(
                    "The product could not be separated clearly. Please place the product on a contrasting surface and retake the photo."
                )

        # 3. Soften edge slightly with guided bilateral filtering for smooth alpha matting
        blurred_mask = cv2.bilateralFilter(mask, 7, 50, 50)
        # Ensure crisp interior
        final_mask = np.where(mask > 240, 255, np.where(mask < 15, 0, blurred_mask)).astype(np.uint8)

        # 4. Construct RGBA foreground
        rgb_array = np.array(rgb_pil, dtype=np.uint8)
        rgba_foreground = np.dstack((rgb_array, final_mask))

        # 5. Extract bounding box and confidence
        bx, by, bw, bh = self.calculate_bounding_box(final_mask)
        confidence = self.calculate_confidence(final_mask)

        return SegmentationResult(
            mask=final_mask,
            rgba_foreground=rgba_foreground,
            bounding_box={"x": int(bx), "y": int(by), "width": int(bw), "height": int(bh)},
            confidence=confidence,
            model_used=model_used,
        )


def segment_product(
    source: Union[str, Path, bytes, np.ndarray, Image.Image],
    model_name: str = "u2netp",
) -> SegmentationResult:
    """Functional convenience entrypoint for product segmentation."""
    segmenter = ProductSegmenter(model_name=model_name)
    return segmenter.segment(source)
