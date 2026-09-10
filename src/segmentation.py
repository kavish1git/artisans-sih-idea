"""Product Segmentation Module (Module 2)
State-of-the-art salient object matting for complex handicrafts:
- Primary: IS-Net (isnet-general-use) for ultra-fine edges, threads, jewelry, and complex contours.
- Supported models: IS-Net, Silueta, U2-Net Standard, and U2-Net Portable (u2netp).
- Intelligent post-processing: connected component noise suppression, morphological gap-closing
  for hanging wires/strings, and adaptive alpha matting.
- Resilient fallback to GrabCut saliency.
"""

from typing import Dict, Any, Tuple, Optional, Union, List
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from src.utils import load_image_safely

# Global model session cache to avoid reloading weights
_MODEL_SESSIONS: Dict[str, Any] = {}


@dataclass
class SegmentationResult:
    """Encapsulates segmentation output metadata."""
    mask: np.ndarray             # 2D uint8 mask [0, 255]
    rgba_foreground: np.ndarray  # 4-channel RGBA array [H, W, 4]
    bounding_box: Dict[str, int] # x, y, width, height
    confidence: float            # 0.0 - 1.0
    model_used: str              # e.g., 'isnet-general-use', 'u2netp'


class SegmentationError(Exception):
    """Raised when segmentation fails to isolate product."""
    pass


class ProductSegmenter:
    """Extracts foreground handicraft products with edge and fine-detail preservation."""

    SUPPORTED_MODELS = [
        "isnet-general-use",
        "silueta",
        "u2netp",
        "u2net",
    ]

    def __init__(
        self,
        model_name: str = "isnet-general-use",
        use_fallback: bool = True,
        enable_alpha_matting: bool = True,
    ):
        self.model_name = model_name
        self.use_fallback = use_fallback
        self.enable_alpha_matting = enable_alpha_matting

    def _get_session(self, model: str):
        """Retrieves or initializes a cached rembg ONNX session."""
        global _MODEL_SESSIONS
        if model in _MODEL_SESSIONS:
            return _MODEL_SESSIONS[model]

        try:
            import rembg
            session = rembg.new_session(model)
            _MODEL_SESSIONS[model] = session
            return session
        except Exception:
            return None

    def _clean_mask_components(self, mask: np.ndarray, detection_mode: str = "auto") -> np.ndarray:
        """
        Filters out stray background noise fragments while preserving
        the primary product cluster, thin hooks, threads, and multipart craft sets.
        Supports: 'auto', 'focused', 'full_set'.
        """
        bin_mask = (mask > 20).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

        if num_labels <= 2:
            return mask

        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = float(np.max(areas))

        clean_bin = np.zeros_like(bin_mask)

        if detection_mode == "full_set":
            # Preserves all components of the craft display/rack (>1% of max area or >20px)
            min_valid = max(20.0, max_area * 0.015)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_valid:
                    clean_bin[labels == i] = 1
        elif detection_mode == "focused":
            # Keeps strictly the largest single focal craft object
            primary_idx = int(np.argmax(areas) + 1)
            clean_bin[labels == primary_idx] = 1
        else:  # "auto"
            # Smart clustering: Retains primary craft and nearby attached pieces (hooks, shells, beads)
            min_valid_area = max(35.0, max_area * 0.035)
            primary_idx = int(np.argmax(areas) + 1)
            primary_center = centroids[primary_idx]
            diag = float(np.sqrt(mask.shape[0] ** 2 + mask.shape[1] ** 2))

            for i in range(1, num_labels):
                comp_area = stats[i, cv2.CC_STAT_AREA]
                dist_to_primary = float(np.linalg.norm(centroids[i] - primary_center))

                if comp_area >= min_valid_area:
                    clean_bin[labels == i] = 1
                elif comp_area >= 15 and dist_to_primary < (0.35 * diag):
                    clean_bin[labels == i] = 1

        # Morphological closing to bridge thin gaps, threads, hooks, and hanging strings
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed_bin = cv2.morphologyEx(clean_bin, cv2.MORPH_CLOSE, kernel)

        # Re-apply softly to original grayscale mask
        cleaned_mask = np.where(closed_bin == 1, mask, 0).astype(np.uint8)
        return cleaned_mask

    def segment_with_dl(
        self,
        rgb_pil: Image.Image,
        model: str,
        detection_mode: str = "auto",
    ) -> Optional[np.ndarray]:
        """Runs deep learning matting with automatic alpha matting and component cleaning."""
        session = self._get_session(model)
        if not session:
            return None

        try:
            import rembg
            rgba_out = rembg.remove(
                rgb_pil,
                session=session,
                only_mask=True,
                post_process_mask=True,
                alpha_matting=self.enable_alpha_matting,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=5,
            )
            mask_np = np.array(rgba_out, dtype=np.uint8)
            if len(mask_np.shape) == 3:
                mask_np = mask_np[:, :, 0]

            # Clean up disconnected noise fragments according to detection mode
            cleaned_mask = self._clean_mask_components(mask_np, detection_mode=detection_mode)
            return cleaned_mask

        except Exception:
            return None

    def segment_with_grabcut(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Robust adaptive GrabCut segmentation as a fallback.
        """
        h, w = bgr_img.shape[:2]
        margin_x = max(5, int(w * 0.05))
        margin_y = max(5, int(h * 0.05))
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(bgr_img, mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
            bin_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
            bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
            return bin_mask
        except Exception:
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            _, bin_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return bin_mask

    def calculate_bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Calculates bounding box [x, y, w, h] around non-zero mask pixels."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            h, w = mask.shape
            return 0, 0, w, h

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
        """Estimates segmentation confidence."""
        h, w = mask.shape
        total_pixels = float(h * w)
        fg_pixels = float(np.sum(mask > 128))
        ratio = fg_pixels / total_pixels

        if ratio < 0.03 or ratio > 0.98:
            return 0.35

        uncertain_pixels = float(np.sum((mask > 15) & (mask < 240)))
        certainty = 1.0 - min(0.5, (uncertain_pixels / (fg_pixels + 1e-5)))

        return round(float(np.clip(certainty * 0.96, 0.4, 0.99)), 2)

    def segment(
        self,
        source: Union[str, Path, bytes, np.ndarray, Image.Image],
        detection_mode: str = "auto",
    ) -> SegmentationResult:
        """
        Executes high-fidelity product segmentation with model cascade fallback.
        """
        if isinstance(source, np.ndarray):
            bgr_img = source
            rgb_pil = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
        else:
            bgr_img, rgb_pil = load_image_safely(source)

        mask: Optional[np.ndarray] = None
        model_used = self.model_name

        # 1. Try requested primary model
        mask = self.segment_with_dl(rgb_pil, self.model_name, detection_mode=detection_mode)

        # 2. Fallback cascade if primary model was unavailable or produced empty mask
        if mask is None or np.sum(mask > 50) < (0.01 * mask.size):
            for fallback_model in self.SUPPORTED_MODELS:
                if fallback_model != self.model_name:
                    mask = self.segment_with_dl(rgb_pil, fallback_model, detection_mode=detection_mode)
                    if mask is not None and np.sum(mask > 50) >= (0.01 * mask.size):
                        model_used = fallback_model
                        break

        # 3. Final heuristic GrabCut fallback
        if mask is None or np.sum(mask > 50) < (0.01 * mask.size):
            if self.use_fallback:
                mask = self.segment_with_grabcut(bgr_img)
                model_used = "grabcut_saliency"
            else:
                raise SegmentationError(
                    "The product could not be separated clearly. Please place the product on a contrasting surface and retake the photo."
                )

        # 4. Refine edge transition with bilateral filtering for smooth alpha matting
        blurred_mask = cv2.bilateralFilter(mask, 7, 50, 50)
        final_mask = np.where(mask > 240, 255, np.where(mask < 15, 0, blurred_mask)).astype(np.uint8)

        # 5. Construct RGBA foreground
        rgb_array = np.array(rgb_pil, dtype=np.uint8)
        rgba_foreground = np.dstack((rgb_array, final_mask))

        # 6. Compute bounding box and confidence
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
    model_name: str = "isnet-general-use",
    detection_mode: str = "auto",
) -> SegmentationResult:
    """Functional convenience entrypoint for product segmentation."""
    segmenter = ProductSegmenter(model_name=model_name)
    return segmenter.segment(source, detection_mode=detection_mode)
