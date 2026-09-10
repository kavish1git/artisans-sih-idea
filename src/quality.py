"""Image Quality Analyzer (Module 1 & 21)
Performs blur detection, brightness/exposure analysis, noise estimation,
composition evaluation, overall scoring (0-100), and generates plain-language
artisan voice feedback.
"""

from typing import Dict, Any, Union, Optional
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from src.utils import load_image_safely


class ImageQualityAnalyzer:
    """Evaluates image quality metrics and generates accessible feedback for artisans."""

    # Thresholds calibrated for smartphone camera captures of handicrafts
    BLUR_THRESHOLD_BAD = 45.0
    BLUR_THRESHOLD_GOOD = 110.0

    BRIGHTNESS_MIN_GOOD = 60.0
    BRIGHTNESS_MAX_GOOD = 225.0

    UNDEREXPOSURE_LIMIT = 0.18  # >18% pixels crushed to black (<15)
    OVEREXPOSURE_LIMIT = 0.15   # >15% pixels blown to pure white (>245)

    NOISE_SIGMA_WARNING = 12.0
    NOISE_SIGMA_BAD = 25.0

    COMPOSITION_MIN_RATIO = 0.18  # Product occupies < 18% of frame
    COMPOSITION_MAX_RATIO = 0.90  # Product occupies > 90% of frame (cropped edges)

    @classmethod
    def analyze_resolution(cls, bgr_img: np.ndarray) -> Dict[str, Any]:
        """Analyzes spatial resolution and megapixels."""
        h, w = bgr_img.shape[:2]
        mp = round((w * h) / 1_000_000.0, 2)

        if w >= 1080 and h >= 1080:
            status = "good"
            res_score = 100
        elif w >= 600 and h >= 600:
            status = "acceptable"
            res_score = 75
        else:
            status = "insufficient"
            res_score = 40

        return {
            "width": int(w),
            "height": int(h),
            "megapixels": float(mp),
            "status": status,
            "score": res_score,
        }

    @classmethod
    def analyze_blur(cls, bgr_img: np.ndarray) -> Dict[str, Any]:
        """
        Estimates image sharpness using variance of Laplacian normalized by contrast.
        Higher values indicate sharp edges; low values indicate motion or defocus blur.
        """
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        std_dev = float(np.std(gray))

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        raw_variance = float(laplacian.var())

        # If image has reasonable contrast, use raw variance;
        # otherwise, scale normalized variance to avoid false positives on low-contrast shots
        if std_dev > 10.0:
            variance = raw_variance
        else:
            # Low contrast: scale variance to equivalent dynamic range
            variance = raw_variance * (35.0 / (std_dev + 1e-3))

        if variance >= cls.BLUR_THRESHOLD_GOOD:
            status = "good"
            score = 100
        elif variance >= cls.BLUR_THRESHOLD_BAD:
            status = "warning"
            score = int(50 + (variance - cls.BLUR_THRESHOLD_BAD) / (cls.BLUR_THRESHOLD_GOOD - cls.BLUR_THRESHOLD_BAD) * 50)
        else:
            status = "bad"
            score = max(5, int((variance / cls.BLUR_THRESHOLD_BAD) * 50))

        return {
            "blur_score": round(variance, 2),
            "blur_status": status,
            "score": score,
        }

    @classmethod
    def _detect_studio_background(cls, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects if an image is on an intentional studio white/off-white background.
        Returns a boolean mask of the foreground (non-background) pixels, or None.
        """
        h, w = gray.shape[:2]
        corners = [
            gray[0, 0], gray[0, w - 1], gray[h - 1, 0], gray[h - 1, w - 1],
            gray[min(5, h - 1), min(5, w - 1)],
            gray[min(5, h - 1), max(0, w - 6)],
            gray[max(0, h - 6), min(5, w - 1)],
            gray[max(0, h - 6), max(0, w - 6)],
        ]
        # Studio background if all sampled corners are very light (>= 238)
        if np.all(np.array(corners) >= 238):
            bg_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(gray.copy(), bg_mask, (0, 0), 0, 6, 6, flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
            cv2.floodFill(gray.copy(), bg_mask, (w - 1, 0), 0, 6, 6, flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
            cv2.floodFill(gray.copy(), bg_mask, (0, h - 1), 0, 6, 6, flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
            cv2.floodFill(gray.copy(), bg_mask, (w - 1, h - 1), 0, 6, 6, flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
            fg_mask = bg_mask[1:-1, 1:-1] == 0
            if np.sum(fg_mask) > 200:
                return fg_mask
        return None

    @classmethod
    def analyze_brightness(cls, bgr_img: np.ndarray) -> Dict[str, Any]:
        """
        Measures perceptual luminance using the L channel in CIE-LAB color space.
        Excludes white studio backdrops to isolate product luminance.
        """
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        fg_mask = cls._detect_studio_background(gray)
        eval_pixels = l_channel[fg_mask] if fg_mask is not None else l_channel

        mean_l = float(np.mean(eval_pixels))
        std_l = float(np.std(eval_pixels))

        if mean_l < cls.BRIGHTNESS_MIN_GOOD:
            status = "too_dark"
            score = max(10, int((mean_l / cls.BRIGHTNESS_MIN_GOOD) * 60))
        elif mean_l > cls.BRIGHTNESS_MAX_GOOD:
            status = "too_bright"
            score = max(10, int((255.0 - mean_l) / (255.0 - cls.BRIGHTNESS_MAX_GOOD) * 60))
        else:
            status = "good"
            score = 100

        return {
            "mean_luminance": round(mean_l, 1),
            "std_luminance": round(std_l, 1),
            "status": status,
            "score": score,
        }

    @classmethod
    def analyze_exposure(cls, bgr_img: np.ndarray) -> Dict[str, Any]:
        """
        Detects severe underexposure (crushed shadows) or overexposure (blown highlights).
        Isolates product region if a studio white backdrop is present.
        """
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        fg_mask = cls._detect_studio_background(gray)

        if fg_mask is not None:
            eval_pixels = gray[fg_mask]
            total_pixels = float(eval_pixels.size)
            # For foreground on white studio background, check true highlight blowout (>250)
            underexposed_ratio = float(np.sum(eval_pixels <= 12) / total_pixels)
            overexposed_ratio = float(np.sum(eval_pixels >= 250) / total_pixels)
        else:
            total_pixels = float(gray.size)
            underexposed_ratio = float(np.sum(gray <= 12) / total_pixels)
            overexposed_ratio = float(np.sum(gray >= 245) / total_pixels)

        if underexposed_ratio > cls.UNDEREXPOSURE_LIMIT:
            status = "severe_underexposure"
            score = max(10, int((1.0 - underexposed_ratio) * 60))
        elif overexposed_ratio > cls.OVEREXPOSURE_LIMIT:
            status = "severe_overexposure"
            score = max(10, int((1.0 - overexposed_ratio) * 60))
        elif underexposed_ratio > 0.08 or overexposed_ratio > 0.06:
            status = "mild_clipping"
            score = 75
        else:
            status = "balanced"
            score = 100

        return {
            "underexposed_ratio": round(underexposed_ratio, 3),
            "overexposed_ratio": round(overexposed_ratio, 3),
            "status": status,
            "score": score,
        }

    @classmethod
    def analyze_noise(cls, bgr_img: np.ndarray) -> Dict[str, Any]:
        """
        Estimates image noise standard deviation via Median Absolute Deviation (MAD)
        of high-frequency residual.
        """
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        residual = gray.astype(np.float32) - blurred.astype(np.float32)

        sigma = float(np.median(np.abs(residual)) / 0.6745)

        if sigma < cls.NOISE_SIGMA_WARNING:
            status = "clean"
            score = 100
        elif sigma < cls.NOISE_SIGMA_BAD:
            status = "moderate"
            score = int(60 + (cls.NOISE_SIGMA_BAD - sigma) / (cls.NOISE_SIGMA_BAD - cls.NOISE_SIGMA_WARNING) * 40)
        else:
            status = "excessive"
            score = max(10, int(60 - (sigma - cls.NOISE_SIGMA_BAD) * 2))

        return {
            "noise_sigma": round(sigma, 2),
            "status": status,
            "score": score,
        }

    @classmethod
    def analyze_composition(cls, bgr_img: np.ndarray) -> Dict[str, Any]:
        """
        Estimates subject occupancy in frame using Otsu saliency & bounding box analysis.
        """
        h, w = bgr_img.shape[:2]
        total_area = float(h * w)

        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(largest_contour)
            product_area = float(bw * bh)
            occupancy_ratio = float(product_area / total_area)
        else:
            occupancy_ratio = 0.5
            bx, by, bw, bh = int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8)

        if occupancy_ratio < cls.COMPOSITION_MIN_RATIO:
            status = "too_small"
            score = 45
        elif occupancy_ratio > cls.COMPOSITION_MAX_RATIO:
            status = "too_close"
            score = 70
        else:
            status = "good"
            score = 100

        return {
            "occupancy_ratio": round(occupancy_ratio, 3),
            "status": status,
            "score": score,
            "estimated_box": {"x": int(bx), "y": int(by), "width": int(bw), "height": int(bh)},
        }

    @classmethod
    def generate_artisan_recommendations(
        cls,
        blur_info: Dict[str, Any],
        brightness_info: Dict[str, Any],
        exposure_info: Dict[str, Any],
        composition_info: Dict[str, Any],
        res_info: Dict[str, Any],
    ) -> str:
        """
        Generates simple, clear, actionable voice recommendations
        designed specifically for marginalized artisans with low digital literacy.
        Priority hierarchy: Lighting/Exposure -> Composition -> Blur -> Resolution.
        """
        # 1. Critical Lighting & Exposure check
        if brightness_info["status"] == "too_dark" or exposure_info["status"] == "severe_underexposure":
            return "Photo is too dark. Please move closer to a window or turn on a light."

        if brightness_info["status"] == "too_bright" or exposure_info["status"] == "severe_overexposure":
            return "Photo has harsh glare. Please shield direct sunlight from the craft."

        # 2. Composition / Framing check
        if composition_info["status"] == "too_small":
            return "Product is too small in the frame. Please move the camera closer."

        if composition_info["status"] == "too_close":
            return "Product is too close to the edges. Please move the camera back slightly."

        # 3. Blur check
        if blur_info["blur_status"] == "bad":
            return "Photo is blurry. Please hold your phone steady and tap to focus."

        if blur_info["blur_status"] == "warning":
            return "Photo is slightly soft. For best results, rest your hands on a stable surface."

        # 4. Resolution check
        if res_info["status"] == "insufficient":
            return "Image resolution is very low. Please ensure phone camera lens is clean."

        return "Product detected clearly! Photo quality is great for cataloging."

    @classmethod
    def evaluate(
        cls, source: Union[str, Path, bytes, np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """
        Runs the full image quality analysis suite on the input image.
        Returns a structured dictionary with scores (0-100) and artisan recommendations.
        """
        if isinstance(source, np.ndarray):
            bgr_img = source
        else:
            bgr_img, _ = load_image_safely(source)

        res_info = cls.analyze_resolution(bgr_img)
        blur_info = cls.analyze_blur(bgr_img)
        brightness_info = cls.analyze_brightness(bgr_img)
        exposure_info = cls.analyze_exposure(bgr_img)
        noise_info = cls.analyze_noise(bgr_img)
        composition_info = cls.analyze_composition(bgr_img)

        # Weighted quality score
        weights = {
            "blur": 0.35,
            "brightness": 0.20,
            "exposure": 0.15,
            "composition": 0.15,
            "resolution": 0.10,
            "noise": 0.05,
        }

        raw_score = (
            blur_info["score"] * weights["blur"]
            + brightness_info["score"] * weights["brightness"]
            + exposure_info["score"] * weights["exposure"]
            + composition_info["score"] * weights["composition"]
            + res_info["score"] * weights["resolution"]
            + noise_info["score"] * weights["noise"]
        )

        # Apply gating penalties for severe disqualifying defects
        if blur_info["blur_status"] == "bad":
            raw_score = min(raw_score, 55.0)
        elif blur_info["blur_status"] == "warning":
            raw_score = min(raw_score, 75.0)

        if brightness_info["status"] in ("too_dark", "too_bright"):
            raw_score = min(raw_score, 60.0)

        if exposure_info["status"] in ("severe_underexposure", "severe_overexposure"):
            raw_score = min(raw_score, 55.0)

        if composition_info["status"] == "too_small":
            raw_score = min(raw_score, 65.0)

        overall_score = max(0, min(100, int(round(raw_score))))

        recommendation = cls.generate_artisan_recommendations(
            blur_info, brightness_info, exposure_info, composition_info, res_info
        )

        return {
            "overall_score": overall_score,
            "recommendation": recommendation,
            "resolution": res_info,
            "blur": blur_info,
            "brightness": brightness_info,
            "exposure": exposure_info,
            "noise": noise_info,
            "composition": composition_info,
        }


def analyze_image_quality(
    source: Union[str, Path, bytes, np.ndarray, Image.Image]
) -> Dict[str, Any]:
    """Convenience functional wrapper for ImageQualityAnalyzer.evaluate."""
    return ImageQualityAnalyzer.evaluate(source)
