"""Image utilities for safe loading, validation, color conversion, and security checks."""

import io
import os
import uuid
from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image, ImageOps

# Maximum allowed file size in bytes (15 MB)
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024

# Dimension boundaries
MIN_IMAGE_DIM = 64
MAX_IMAGE_DIM = 8192

# Supported formats
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

# Recognized magic bytes
MAGIC_NUMBERS = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"RIFF": "webp",  # RIFF....WEBP
    b"BM": "bmp",
    b"II*\x00": "tiff",  # Little-endian TIFF
    b"MM\x00*": "tiff",  # Big-endian TIFF
}


class ImageValidationError(Exception):
    """Exception raised when an image fails security or format validation."""
    pass


def validate_image_bytes(data: bytes, filename: Optional[str] = None) -> str:
    """
    Validates raw byte buffer for size, magic bytes, and format compatibility.
    
    Args:
        data: Raw image bytes.
        filename: Optional source filename to check extension.
        
    Returns:
        Detected format string ('jpeg', 'png', etc.).
        
    Raises:
        ImageValidationError: If file is too large, empty, or has an invalid format.
    """
    if not data or len(data) == 0:
        raise ImageValidationError("The uploaded file is empty.")

    if len(data) > MAX_FILE_SIZE_BYTES:
        size_mb = len(data) / (1024 * 1024)
        raise ImageValidationError(
            f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of 15 MB."
        )

    # Magic byte check
    detected_format = None
    for magic, fmt in MAGIC_NUMBERS.items():
        if data.startswith(magic):
            if fmt == "webp" and len(data) > 12 and data[8:12] != b"WEBP":
                continue
            detected_format = fmt
            break

    if not detected_format:
        raise ImageValidationError(
            "Unsupported or corrupted file format. Only JPEG, PNG, WebP, BMP, and TIFF are supported."
        )

    if filename:
        ext = Path(filename).suffix.lower()
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise ImageValidationError(f"File extension '{ext}' is not supported.")

    return detected_format


def load_image_safely(
    source: Union[str, Path, bytes, io.BytesIO, Image.Image]
) -> Tuple[np.ndarray, Image.Image]:
    """
    Loads an image safely from path, bytes, stream, or PIL Image.
    Applies EXIF orientation normalization, checks dimensions, and converts to:
    1. BGR numpy ndarray (OpenCV format)
    2. RGB PIL Image
    
    Args:
        source: File path, bytes, BytesIO, or PIL Image.
        
    Returns:
        Tuple of (cv2_bgr_array, pil_rgb_image)
        
    Raises:
        ImageValidationError: If validation fails or image cannot be decoded.
    """
    pil_img: Optional[Image.Image] = None

    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise ImageValidationError(f"File not found: {source}")
            with open(path, "rb") as f:
                data = f.read()
            validate_image_bytes(data, filename=path.name)
            pil_img = Image.open(io.BytesIO(data))
        elif isinstance(source, bytes):
            validate_image_bytes(source)
            pil_img = Image.open(io.BytesIO(source))
        elif isinstance(source, io.BytesIO):
            source.seek(0)
            data = source.read()
            validate_image_bytes(data)
            pil_img = Image.open(io.BytesIO(data))
        elif isinstance(source, Image.Image):
            pil_img = source.copy()
        else:
            raise ImageValidationError(f"Unsupported image input type: {type(source)}")

        # Automatically transpose based on EXIF tag (critical for smartphone photos)
        try:
            pil_img = ImageOps.exif_transpose(pil_img)
        except Exception:
            pass  # Fall back gracefully if EXIF parsing fails

        # Dimension checks
        w, h = pil_img.size
        if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
            raise ImageValidationError(
                f"Image resolution ({w}x{h}) is too small. Minimum required dimension is {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}px."
            )
        if w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM:
            raise ImageValidationError(
                f"Image resolution ({w}x{h}) exceeds maximum supported size ({MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}px)."
            )

        # Convert colorspace to standard RGB for PIL
        if pil_img.mode in ("RGBA", "LA"):
            # Keep alpha or convert safely
            rgb_img = pil_img.convert("RGB")
        elif pil_img.mode == "CMYK":
            rgb_img = pil_img.convert("RGB")
        elif pil_img.mode in ("L", "1"):
            rgb_img = pil_img.convert("RGB")
        elif pil_img.mode == "RGB":
            rgb_img = pil_img
        else:
            rgb_img = pil_img.convert("RGB")

        # Convert to OpenCV BGR array (uint8)
        rgb_array = np.array(rgb_img, dtype=np.uint8)
        # RGB to BGR for cv2
        bgr_array = rgb_array[:, :, ::-1].copy()

        return bgr_array, rgb_img

    except ImageValidationError:
        raise
    except Exception as e:
        raise ImageValidationError(f"Failed to read or decode image: {str(e)}") from e


def generate_unique_filename(extension: str = ".jpg") -> str:
    """Generates a secure, non-colliding random filename."""
    if not extension.startswith("."):
        extension = "." + extension
    return f"{uuid.uuid4().hex}{extension}"
