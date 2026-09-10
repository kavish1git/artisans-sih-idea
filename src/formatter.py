"""Marketplace Image Formatter (Module 10)
Standardizes images for e-commerce publishing:
- Default 1080 x 1080 resolution
- High-quality export: JPEG (default quality 90), WebP, PNG
- Metadata stripping for privacy and payload compression
- High-fidelity resampling using Lanczos interpolation.
"""

from typing import Union, Tuple, Optional
from pathlib import Path
import io
from PIL import Image


class MarketplaceFormatter:
    """Formats processed images for e-commerce catalog standards."""

    DEFAULT_WIDTH = 1080
    DEFAULT_HEIGHT = 1080
    DEFAULT_FORMAT = "JPEG"
    DEFAULT_QUALITY = 90

    @classmethod
    def format_image(
        cls,
        image: Image.Image,
        target_width: int = DEFAULT_WIDTH,
        target_height: int = DEFAULT_HEIGHT,
        export_format: str = DEFAULT_FORMAT,
        quality: int = DEFAULT_QUALITY,
    ) -> Image.Image:
        """
        Resamples image to target e-commerce dimensions and converts mode cleanly.
        
        Args:
            image: PIL Image.
            target_width: Target width in px (default 1080).
            target_height: Target height in px (default 1080).
            export_format: 'JPEG', 'PNG', or 'WEBP'.
            quality: Compression quality (1-100, default 90).
            
        Returns:
            Resampled PIL Image formatted for export.
        """
        fmt = export_format.upper()
        if fmt == "JPG":
            fmt = "JPEG"

        # High quality Lanczos downsampling/upsampling
        if image.size != (target_width, target_height):
            resampled = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        else:
            resampled = image.copy()

        # Mode adaptation based on target format
        if fmt == "JPEG":
            if resampled.mode in ("RGBA", "LA"):
                # Composite onto pure white if converting transparent RGBA to JPEG
                bg = Image.new("RGB", resampled.size, (255, 255, 255))
                bg.paste(resampled, (0, 0), resampled)
                resampled = bg
            elif resampled.mode != "RGB":
                resampled = resampled.convert("RGB")
        elif fmt in ("PNG", "WEBP"):
            # PNG and WEBP support transparency
            if resampled.mode not in ("RGB", "RGBA"):
                resampled = resampled.convert("RGB")

        return resampled

    @classmethod
    def export_bytes(
        cls,
        image: Image.Image,
        export_format: str = DEFAULT_FORMAT,
        quality: int = DEFAULT_QUALITY,
        strip_metadata: bool = True,
    ) -> bytes:
        """Encodes image to byte buffer with stripped metadata."""
        fmt = export_format.upper()
        if fmt == "JPG":
            fmt = "JPEG"

        buffer = io.BytesIO()
        save_kwargs = {}

        if fmt in ("JPEG", "WEBP"):
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True
        elif fmt == "PNG":
            save_kwargs["optimize"] = True

        # Stripping metadata by saving without original info dictionary
        image.save(buffer, format=fmt, **save_kwargs)
        return buffer.getvalue()

    @classmethod
    def save_to_file(
        cls,
        image: Image.Image,
        output_path: Union[str, Path],
        target_width: int = DEFAULT_WIDTH,
        target_height: int = DEFAULT_HEIGHT,
        export_format: Optional[str] = None,
        quality: int = DEFAULT_QUALITY,
    ) -> Path:
        """Formats and saves image to filesystem."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not export_format:
            ext = output_path.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                export_format = "JPEG"
            elif ext == ".png":
                export_format = "PNG"
            elif ext == ".webp":
                export_format = "WEBP"
            else:
                export_format = cls.DEFAULT_FORMAT

        formatted = cls.format_image(
            image,
            target_width=target_width,
            target_height=target_height,
            export_format=export_format,
            quality=quality,
        )

        data = cls.export_bytes(formatted, export_format=export_format, quality=quality)
        with open(output_path, "wb") as f:
            f.write(data)

        return output_path


def format_marketplace_image(
    image: Image.Image,
    width: int = 1080,
    height: int = 1080,
    export_format: str = "JPEG",
    quality: int = 90,
) -> Image.Image:
    """Convenience helper for marketplace formatting."""
    return MarketplaceFormatter.format_image(
        image, target_width=width, target_height=height, export_format=export_format, quality=quality
    )
