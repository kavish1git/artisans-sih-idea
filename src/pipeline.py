"""Complete Computer Vision Pipeline Orchestrator (Modules 11, 12, 17)
Fully Automatic AI Pipeline for SIH26090:
1. Image validation & pre-flight quality check
2. Salient product segmentation with model cascade (IS-Net -> Silueta -> U2Net -> GrabCut)
3. Quality verification & retake guardrails (prevents empty/failed cutouts)
4. Indian handicraft product detection & taxonomy classification (non-hallucinating)
5. Objective visual attributes extraction (colors, shape, orientation, pattern, texture)
6. Adaptive lighting correction & Shades-of-Gray color constancy
7. Proportional aspect ratio alignment & smart centering (1080x1080)
8. Intelligent grounding shadow synthesis (3D contact vs natural ambient vs flat lay)
9. Contrast-aware marketplace backdrop selection (prevents white-on-white edge bleed)
10. Final catalog verification (before/after quality delta)
11. Structured Product Profile generation matching SIH multi-agent contract.
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
from src.shadow import apply_grounding_shadow, ShadowHandler
from src.marketplace import MarketplaceGenerator, BACKGROUND_PALETTES
from src.formatter import MarketplaceFormatter
from src.detector import detect_product
from src.attributes import extract_attributes


class PipelineError(Exception):
    """Exception raised when end-to-end pipeline detects critical validation failure."""
    pass


class ProductImagePipeline:
    """Orchestrates fully automatic production computer vision pipeline."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "output",
        model_name: str = "isnet-general-use",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segmenter = ProductSegmenter(model_name=model_name)
        self.remover = BackgroundRemover(segmenter=self.segmenter)

    def process(
        self,
        source: Union[str, Path, bytes, np.ndarray, Image.Image],
        background: str = "auto",
        aspect_ratio: str = "auto",
        enhancement: str = "auto",
        shadow_mode: str = "auto",
        target_dim: int = 1080,
        output_format: str = "JPEG",
        model_name: Optional[str] = None,
        alpha_matting: bool = True,
        detection_mode: str = "auto",
        save_files: bool = True,
        skip_quality_gate: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes fully automatic production cataloging pipeline from a single photo.
        """
        start_time = time.perf_counter()
        warnings: List[str] = []

        # 1. Intelligent Pre-flight Quality & Identifiability Gate
        if not skip_quality_gate:
            from src.quality_gate import evaluate_image_gate
            gate_decision = evaluate_image_gate(source)

            if not gate_decision["accepted"]:
                proc_time = round((time.perf_counter() - start_time) * 1000, 1)
                primary_issue = gate_decision["issues"][0]["message"] if gate_decision["issues"] else "Photo needs to be retaken."
                return {
                    "status": "needs_retake",
                    "accepted": False,
                    "action": "RETAKE_PHOTO",
                    "reason": primary_issue,
                    "voice_prompt": gate_decision["voice_prompt"],
                    "quality": gate_decision["quality"],
                    "issues": gate_decision["issues"],
                    "retake_instruction": gate_decision["retake_instruction"],
                    "suggestions": gate_decision["suggestions"],
                    "quality_score": gate_decision["quality"]["overall_score"],
                    "product_detection": gate_decision.get("product_detection", {}),
                    "processing_time_ms": proc_time,
                }

        # Safe Load for Pipeline Processing
        bgr_orig, rgb_pil = load_image_safely(source)
        rgb_np = np.array(rgb_pil)
        quality_before = analyze_image_quality(bgr_orig)

        # 2. Product Segmentation with Automatic Cascade
        # Configure model if specified, otherwise rely on default IS-Net with fallbacks
        if model_name and model_name not in ("auto", "default") and model_name != self.segmenter.model_name:
            self.segmenter.model_name = model_name
        self.segmenter.enable_alpha_matting = alpha_matting

        try:
            transparent_pil, seg_res = self.remover.remove_background(rgb_pil, detection_mode=detection_mode)
        except SegmentationError as e:
            proc_time = round((time.perf_counter() - start_time) * 1000, 1)
            return {
                "status": "needs_retake",
                "reason": "The main product could not be separated clearly from the background.",
                "voice_prompt": "Please place the product on a clear surface and take the photo again.",
                "quality_before": quality_before,
                "processing_time_ms": proc_time,
            }

        # Retake check 2: Empty or failed mask
        fg_pixel_count = int(np.sum(seg_res.mask > 40))
        total_pixels = seg_res.mask.size
        product_area_pct = round(float((fg_pixel_count / total_pixels) * 100.0), 1)

        if product_area_pct < 0.8 or product_area_pct > 98.8:
            proc_time = round((time.perf_counter() - start_time) * 1000, 1)
            return {
                "status": "needs_retake",
                "reason": "The main product could not be separated clearly from the background.",
                "voice_prompt": "Please place the product on a clear surface and take the photo again.",
                "quality_before": quality_before,
                "processing_time_ms": proc_time,
            }

        # 3. Automatic Product Detection & Indian Handicraft Recognition
        product_detection = detect_product(
            rgb_np,
            mask=seg_res.mask,
            bounding_box=seg_res.bounding_box,
        )

        # 4. Objective Visual Attributes Extraction
        visual_attrs = extract_attributes(
            rgb_np,
            mask=seg_res.mask,
            bounding_box=seg_res.bounding_box,
        )

        # 5. Tone & Color Restoration (on foreground pixels)
        lighting_fixed = correct_lighting(transparent_pil, mask=seg_res.mask)
        wb_fixed = correct_white_balance(lighting_fixed, mask=seg_res.mask)

        # 6. Proportional Aspect Ratio Alignment & Smart Centering
        resolved_ratio = "1:1" if aspect_ratio in ("auto", "default") else aspect_ratio
        centered_pil, final_crop_box = crop_and_center_product(
            wb_fixed,
            mask_or_bbox=seg_res.bounding_box,
            aspect_ratio=resolved_ratio,
            padding=0.08,
        )

        # 7. Intelligent Grounding Shadow Handling
        if shadow_mode == "auto":
            resolved_shadow, shadow_intensity = ShadowHandler.select_automatic_shadow_mode(
                centered_pil,
                category=product_detection["category"],
                shape=visual_attrs["shape"],
            )
        else:
            resolved_shadow = shadow_mode
            shadow_intensity = 0.30

        if resolved_shadow != "none" and shadow_intensity > 0.01:
            grounded_pil = apply_grounding_shadow(
                centered_pil,
                bbox=final_crop_box,
                mode=resolved_shadow,
                intensity=shadow_intensity,
            )
            shadow_applied_desc = f"{resolved_shadow}_shadow"
        else:
            grounded_pil = centered_pil
            shadow_applied_desc = "none"

        # 8. Intelligent Contrast-Aware Background Decision
        if background == "auto":
            resolved_bg = MarketplaceGenerator.select_automatic_background(grounded_pil)
        else:
            resolved_bg = background

        # 9. Clean Neutral Backdrop Generation & Conservative Enhancement
        marketplace_pil = MarketplaceGenerator.enhance_product(
            grounded_pil,
            sharpness="medium" if enhancement in ("auto", "medium") else "low",
            contrast="auto",
            saturation="low",
            denoise=True,
        )
        final_rgb = MarketplaceGenerator.apply_background(marketplace_pil, background_type=resolved_bg)

        # 10. Marketplace Resampling & Standardization (1080x1080)
        final_export_fmt = "PNG" if resolved_bg == "transparent" else output_format
        formatted_img = MarketplaceFormatter.format_image(
            final_rgb,
            target_width=target_dim,
            target_height=target_dim,
            export_format=final_export_fmt,
            quality=90,
        )

        # 11. Post-Processing Quality Verification (After)
        bgr_after = cv2.cvtColor(np.array(formatted_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        quality_after = analyze_image_quality(bgr_after)
        improvement = max(0, quality_after["overall_score"] - quality_before["overall_score"])

        # 12. File Persistence
        file_id = uuid.uuid4().hex[:12]
        ext = "png" if final_export_fmt.upper() == "PNG" else "jpg"
        orig_filename = f"orig_{file_id}.jpg"
        catalog_filename = f"catalog_{file_id}.{ext}"
        trans_filename = f"transparent_{file_id}.png"

        orig_path = self.output_dir / orig_filename
        out_path = self.output_dir / catalog_filename
        trans_path = self.output_dir / trans_filename

        if save_files:
            # Save original for before/after comparison
            rgb_pil.convert("RGB").save(str(orig_path), format="JPEG", quality=88)
            # Save processed catalog listing
            MarketplaceFormatter.save_to_file(
                formatted_img,
                out_path,
                target_width=target_dim,
                target_height=target_dim,
                export_format=final_export_fmt,
                quality=90,
            )
            # Save transparent cutout
            centered_pil.save(str(trans_path), format="PNG", optimize=True)

        proc_time = round((time.perf_counter() - start_time) * 1000, 1)

        # 13. Construct Complete Structured Product Profile
        bg_hex_map = {
            "white": "#FFFFFF",
            "off-white": "#F8F9FA",
            "light-gray": "#F0F0F2",
            "transparent": "transparent",
        }
        chosen_bg_desc = f"{resolved_bg} ({bg_hex_map.get(resolved_bg, '#F8F9FA')})"

        # Craft human-friendly spoken recommendation
        voice_text = (
            f"{product_detection['display_text']}. "
            f"Image cleaned, lighting improved, and prepared for marketplace."
        )

        # Post-Processing Quality Verification Check
        processing_status = "SUCCESS" if (quality_after["overall_score"] >= 45 and product_area_pct >= 4.0) else "FAILED"

        return {
            "status": "success",
            "accepted": True,
            "action": "PROCESS_IMAGE",
            "processing_status": processing_status,
            "product": {
                "name": product_detection["name"],
                "category": product_detection["category"],
                "subcategory": product_detection.get("subcategory", ""),
                "craft_type": product_detection.get("craft_type", ""),
                "confidence": product_detection["confidence"],
                "confidence_label": product_detection["confidence_label"],
                "display_text": product_detection["display_text"],
            },
            "product_detection": product_detection,
            "visual_attributes": {
                "colors": visual_attrs["dominant_colors"],
                "palette_hex": visual_attrs["palette_hex"],
                "shape": visual_attrs["shape"],
                "pattern": visual_attrs["pattern"],
                "orientation": visual_attrs["orientation"],
                "texture": visual_attrs["texture"],
            },
            "geometry": {
                "bounding_box": seg_res.bounding_box,
                "product_area_percentage": product_area_pct,
            },
            "image": {
                "original": str(orig_path) if save_files else None,
                "processed": str(out_path) if save_files else None,
                "transparent": str(trans_path) if save_files else None,
                "width": target_dim,
                "height": target_dim,
            },
            "quality": {
                "before": quality_before["overall_score"],
                "after": quality_after["overall_score"],
                "improvement": improvement,
                "metrics_before": quality_before,
                "metrics_after": quality_after,
            },
            "issues": [],
            "retake_instruction": "Photo passed quality gate and catalog image created successfully.",
            "suggestions": [],
            "processing": {
                "time_ms": proc_time,
                "segmentation_engine": seg_res.model_used,
                "background_chosen": chosen_bg_desc,
                "shadow_applied": shadow_applied_desc,
                "aspect_ratio_chosen": resolved_ratio,
            },
            "voice_prompt": voice_text,
            "warnings": warnings,
            # Top-level backward compatibility aliases for existing test suites
            "output_image": str(out_path) if save_files else None,
            "transparent_image": str(trans_path) if save_files else None,
            "before_score": quality_before["overall_score"],
            "after_score": quality_after["overall_score"],
            "improvement": improvement,
            "dominant_colors": visual_attrs["palette_hex"],
            "bounding_box": seg_res.bounding_box,
            "processing_time_ms": proc_time,
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
    image: Optional[Union[str, Path, bytes, Image.Image]] = None,
    image_path: Optional[Union[str, Path, bytes, Image.Image]] = None,
    background: str = "auto",
    aspect_ratio: str = "auto",
    enhancement: str = "auto",
    shadow_mode: str = "auto",
    output_format: str = "JPEG",
    target_dim: int = 1080,
    model_name: str = "auto",
    alpha_matting: bool = True,
    detection_mode: str = "auto",
    skip_quality_gate: bool = False,
) -> Dict[str, Any]:
    """
    Master functional entrypoint for fully automatic image processing.
    Accepts ONLY the input image (via `image` or `image_path`).
    """
    target = image if image is not None else image_path
    if target is None:
        raise ValueError("Either `image` or `image_path` must be provided.")
    pipeline = get_pipeline()
    return pipeline.process(
        source=target,
        background=background,
        aspect_ratio=aspect_ratio,
        enhancement=enhancement,
        shadow_mode=shadow_mode,
        target_dim=target_dim,
        output_format=output_format,
        model_name=model_name,
        alpha_matting=alpha_matting,
        detection_mode=detection_mode,
        skip_quality_gate=skip_quality_gate,
    )

