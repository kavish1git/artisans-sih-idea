"""Product Cropping and Centering Module (Module 7)
Centers product with configurable padding and supports standard e-commerce aspect ratios:
1:1 (default), 4:5, 3:4, 16:9.
"""

from typing import Tuple, Dict, Any, Union, Optional
import cv2
import numpy as np
from PIL import Image


ASPECT_RATIOS = {
    "1:1": (1.0, 1.0),
    "4:5": (4.0, 5.0),
    "3:4": (3.0, 4.0),
    "16:9": (16.0, 9.0),
}


class ProductCropper:
    """Handles smart bounding-box framing, centering, and aspect-ratio alignment."""

    def __init__(self, default_padding: float = 0.08):
        self.default_padding = default_padding

    @staticmethod
    def parse_aspect_ratio(ratio_str: str) -> Tuple[float, float]:
        """Parses aspect ratio string like '1:1', '4:5', '3:4', '16:9'."""
        if ratio_str in ASPECT_RATIOS:
            return ASPECT_RATIOS[ratio_str]
        try:
            parts = ratio_str.split(":")
            return float(parts[0]), float(parts[1])
        except Exception:
            return 1.0, 1.0

    def crop_and_center(
        self,
        image: Union[Image.Image, np.ndarray],
        mask_or_bbox: Optional[Union[np.ndarray, Dict[str, int]]] = None,
        aspect_ratio: str = "1:1",
        padding: Optional[float] = None,
    ) -> Tuple[Image.Image, Dict[str, int]]:
        """
        Crops and centers the product on an expanded canvas adhering to the aspect ratio.
        
        Args:
            image: RGBA PIL Image or numpy array.
            mask_or_bbox: Either a 2D binary mask or dict {'x': x, 'y': y, 'width': w, 'height': h}.
            aspect_ratio: Target aspect ratio ('1:1', '4:5', '3:4', '16:9').
            padding: Margin fraction around product (default 0.08 = 8%).
            
        Returns:
            Tuple of (centered_pil_image, final_crop_bbox)
        """
        if isinstance(image, np.ndarray):
            if image.shape[2] == 4:
                pil_img = Image.fromarray(image, mode="RGBA")
            else:
                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = image

        w_img, h_img = pil_img.size
        pad_frac = padding if padding is not None else self.default_padding

        # 1. Determine bounding box
        bx, by, bw, bh = 0, 0, w_img, h_img
        if isinstance(mask_or_bbox, dict):
            bx = mask_or_bbox.get("x", 0)
            by = mask_or_bbox.get("y", 0)
            bw = mask_or_bbox.get("width", w_img)
            bh = mask_or_bbox.get("height", h_img)
        elif isinstance(mask_or_bbox, np.ndarray) and mask_or_bbox.ndim == 2:
            contours, _ = cv2.findContours(mask_or_bbox, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                valid = [c for c in contours if cv2.contourArea(c) > 50]
                if valid:
                    min_x, min_y = w_img, h_img
                    max_x, max_y = 0, 0
                    for c in valid:
                        x, y, w, h = cv2.boundingRect(c)
                        min_x = min(min_x, x)
                        min_y = min(min_y, y)
                        max_x = max(max_x, x + w)
                        max_y = max(max_y, y + h)
                    bx, by, bw, bh = min_x, min_y, max_x - min_x, max_y - min_y
        elif pil_img.mode == "RGBA":
            # Extract bbox from alpha channel
            alpha = np.array(pil_img.split()[-1])
            y_indices, x_indices = np.where(alpha > 10)
            if len(y_indices) > 0:
                bx = int(np.min(x_indices))
                by = int(np.min(y_indices))
                bw = int(np.max(x_indices) - bx + 1)
                bh = int(np.max(y_indices) - by + 1)

        # 2. Extract product crop with safe bounds
        crop_box = (
            max(0, bx),
            max(0, by),
            min(w_img, bx + bw),
            min(h_img, by + bh),
        )
        product_cropped = pil_img.crop(crop_box)
        prod_w, prod_h = product_cropped.size

        # 3. Calculate target canvas dimension with padding
        # Padded product dimension
        padded_w = prod_w * (1.0 + 2 * pad_frac)
        padded_h = prod_h * (1.0 + 2 * pad_frac)

        target_rw, target_rh = self.parse_aspect_ratio(aspect_ratio)
        target_ratio = target_rw / target_rh

        current_ratio = padded_w / padded_h

        if current_ratio > target_ratio:
            # Width is constraining; expand height
            canvas_w = int(round(padded_w))
            canvas_h = int(round(padded_w / target_ratio))
        else:
            # Height is constraining; expand width
            canvas_h = int(round(padded_h))
            canvas_w = int(round(padded_h * target_ratio))

        # Ensure canvas dimensions are at least equal to product
        canvas_w = max(canvas_w, prod_w)
        canvas_h = max(canvas_h, prod_h)

        # 4. Create blank transparent canvas and paste product in exact center
        mode = "RGBA" if pil_img.mode == "RGBA" else "RGB"
        bg_color = (0, 0, 0, 0) if mode == "RGBA" else (255, 255, 255)
        canvas = Image.new(mode, (canvas_w, canvas_h), bg_color)

        paste_x = (canvas_w - prod_w) // 2
        paste_y = (canvas_h - prod_h) // 2

        if mode == "RGBA":
            canvas.paste(product_cropped, (paste_x, paste_y), product_cropped)
        else:
            canvas.paste(product_cropped, (paste_x, paste_y))

        final_box = {
            "x": paste_x,
            "y": paste_y,
            "width": prod_w,
            "height": prod_h,
            "canvas_width": canvas_w,
            "canvas_height": canvas_h,
        }

        return canvas, final_box


def crop_and_center_product(
    image: Union[Image.Image, np.ndarray],
    mask_or_bbox: Optional[Union[np.ndarray, Dict[str, int]]] = None,
    aspect_ratio: str = "1:1",
    padding: float = 0.08,
) -> Tuple[Image.Image, Dict[str, int]]:
    """Functional helper for smart product cropping and centering."""
    cropper = ProductCropper(default_padding=padding)
    return cropper.crop_and_center(image, mask_or_bbox, aspect_ratio, padding)
