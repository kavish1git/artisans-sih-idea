"""Complete Computer Vision Pipeline Orchestrator (Modules 11 & 12)
Transforms poor smartphone photos into professional marketplace catalog images:
1. Quality Analysis (Before)
2. Product Segmentation (U2-Net / GrabCut fallback)
3. Edge-Preserving Background Removal (Transparent PNG)
4. Adaptive Lighting & Exposure Correction
5. White Balance & Color Constancy
6. Smart Bounding Box Centering & Aspect Ratio Alignment (1:1, 4:5, 3:4, 16:9)
7. Grounding Shadow Synthesis (Professional vs. Natural)
8. Clean Backdrop Generation (White, Off-White, Light Gray)
9. Conservative Enhancement & Formatting (1080x1080 JPEG/WebP)
10. Quality Verification (Before vs. After Comparison & Anomaly Safeguards)
11. Dominant Color & Visual Metadata Extraction
"""

import time
import uuid
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
import cv2
import numpy as np
from PIL import Image

from src.utils import load_image_safely, generate_unique_filename
from src.quality import analyze_image_quality
from src.segmentation import ProductSegmenter, SegmentationError
from src.background import BackgroundRemover
from src.lighting import correct_lighting
from src.white_balance import correct_white_balance
from src.crop import crop_and_center_product
from src.shadow import apply_grounding_shadow
from src.marketplace import MarketplaceGenerator
from src.formatter import MarketplaceFormatter


class PipelineError(Exception):
    """Exception raised when end-to-end pipeline detects critical validation failure."""
    pass


class ProductImagePipeline:
    """Orchestrates end-to-end production computer vision pipeline."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "output",
        model_name: str = "u2netp",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segmenter = ProductSegmenter(model_name=model_name)
        self.remover = BackgroundRemover(segmenter=self.segmenter)

    @staticmethod
    def extract_dominant_colors(rgba_img: Image.Image, k: int = 4) -> List[str]:
        """
        Extracts dominant product colors (hex codes) from non-transparent foreground pixels.
        """
        arr = np.array(rgba_img)
        if arr.shape[2] != 4:
            return ["#808080"]

        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3]
        fg_pixels = rgb[alpha > 80]

        if len(fg_pixels) < 20:
            return ["#808080"]

        # Subsample for speed
        if len(fg_pixels) > 5000:
            indices = np.random.choice(len(fg_pixels), 5000, replace=False)
            samples = fg_pixels[indices].astype(np.float32)
        else:
            samples = fg_pixels.astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

        counts = np.bincount(labels.flatten())
        sorted_indices = np.argsort(-counts)

        hex_colors = []
        for idx in sorted_indices:
            c = centers[idx].astype(int)
            hex_code = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
            hex_colors.append(hex_code)

        return hex_colors

    def process(
        self,
        source: Union[str, Path, bytes, np.ndarray, Image.Image],
        background: str = "white",
        aspect_ratio: str = "1:1",
        enhancement: str = "auto",
        shadow_mode: str = "professional",
        target_dim: int = 1080,
        output_format: str = "JPEG",
        save_files: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes complete production cataloging pipeline.
        """
        start_time = time.perf_counter()
        warnings: List[str] = []

        # 1. Safe Load & Initial Quality Analysis (Before)
        bgr_orig, rgb_pil = load_image_safely(source)
        quality_before = analyze_image_quality(bgr_orig)

        if quality_before["overall_score"] < 40:
            warnings.append(
                f"Low input quality ({quality_before['overall_score']}/100): {quality_before['recommendation']}"
            )

        # 2. Product Segmentation & Background Removal
        try:
            transparent_pil, seg_res = self.remover.remove_background(rgb_pil)
        except SegmentationError as e:
            # Safe failure fallback: Return original image with clear warning
            proc_time = round((time.perf_counter() - start_time) * 1000, 1)
            return {
                "status": "warning",
                "message": str(e),
                "quality_before": quality_before,
                "quality_after": quality_before,
                "improvement": 0,
                "bounding_box": {},
                "dominant_colors": [],
                "processing_time_ms": proc_time,
                "warnings": [str(e)],
            }

        # 3. Tone & Color Restoration (on foreground pixels)
        # Lighting correction (CLAHE + Gamma)
        lighting_fixed = correct_lighting(transparent_pil, mask=seg_res.mask)
        # White balance correction (Shades-of-Gray constancy)
        wb_fixed = correct_white_balance(lighting_fixed, mask=seg_res.mask)

        # 4. Smart Cropping & Centering to Target Aspect Ratio
        centered_pil, final_crop_box = crop_and_center_product(
            wb_fixed,
            mask_or_bbox=seg_res.bounding_box,
            aspect_ratio=aspect_ratio,
            padding=0.08,
        )

        # 5. Grounding Shadow Generation
        if shadow_mode.lower() != "none":
            grounded_pil = apply_grounding_shadow(
                centered_pil,
                bbox=final_crop_box,
                mode=shadow_mode,
                intensity=0.32,
            )
        else:
            grounded_pil = centered_pil

        # 6. Clean Neutral Backdrop Generation & Conservative Enhancement
        marketplace_pil = MarketplaceGenerator.enhance_product(
            grounded_pil,
            sharpness="medium" if enhancement in ("auto", "medium") else "low",
            contrast="auto",
            saturation="low",
            denoise=True,
        )
        final_rgb = MarketplaceGenerator.apply_background(marketplace_pil, background_type=background)

        # 7. Marketplace Resampling & Standardization (1080x1080)
        formatted_img = MarketplaceFormatter.format_image(
            final_rgb,
            target_width=target_dim,
            target_height=target_dim,
            export_format=output_format,
            quality=90,
        )

        # 8. Post-Processing Quality Verification (After)
        # Convert formatted PIL to BGR array for quality analyzer
        bgr_after = cv2.cvtColor(np.array(formatted_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        quality_after = analyze_image_quality(bgr_after)

        # Sanity check: Ensure foreground wasn't obliterated
        after_alpha_check = np.array(centered_pil.split()[-1])
        if np.sum(after_alpha_check > 50) < 500:
            warnings.append("Warning: Segmented product area is suspiciously small.")

        improvement = max(0, quality_after["overall_score"] - quality_before["overall_score"])

        # 9. Extract Dominant Colors for Catalog Filtering
        dominant_colors = self.extract_dominant_colors(wb_fixed, k=4)

        # 10. File Persistence
        file_id = uuid.uuid4().hex[:12]
        output_filename = f"catalog_{file_id}.{output_format.lower() if output_format.lower() != 'jpeg' else 'jpg'}"
        trans_filename = f"transparent_{file_id}.png"

        out_path = self.output_dir / output_filename
        trans_path = self.output_dir / trans_filename

        if save_files:
            MarketplaceFormatter.save_to_file(
                formatted_img,
                out_path,
                target_width=target_dim,
                target_height=target_dim,
                export_format=output_format,
                quality=90,
            )
            centered_pil.save(str(trans_path), format="PNG", optimize=True)

        proc_time = round((time.perf_counter() - start_time) * 1000, 1)

        return {
            "status": "success",
            "output_image": str(out_path) if save_files else None,
            "transparent_image": str(trans_path) if save_files else None,
            "quality_before": quality_before,
            "quality_after": quality_after,
            "before_score": quality_before["overall_score"],
            "after_score": quality_after["overall_score"],
            "improvement": improvement,
            "bounding_box": seg_res.bounding_box,
            "dominant_colors": dominant_colors,
            "aspect_ratio": aspect_ratio,
            "background": background,
            "processing_time_ms": proc_time,
            "recommendation": quality_before["recommendation"],
            "warnings": warnings,
        }


# Global singleton pipeline instance
_default_pipeline: Optional[ProductImagePipeline] = None


def get_pipeline() -> ProductImagePipeline:
    """Returns singleton pipeline instance."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = ProductImagePipeline()
    return _default_pipeline


def process_product_image(
    image_path: Union[str, Path, bytes, Image.Image],
    background: str = "white",
    aspect_ratio: str = "1:1",
    enhancement: str = "auto",
    shadow_mode: str = "professional",
    output_format: str = "JPEG",
    target_dim: int = 1080,
) -> Dict[str, Any]:
    """Exposed functional API matching Module 12 specification."""
    pipeline = get_pipeline()
    return pipeline.process(
        source=image_path,
        background=background,
        aspect_ratio=aspect_ratio,
        enhancement=enhancement,
        shadow_mode=shadow_mode,
        target_dim=target_dim,
        output_format=output_format,
    )
