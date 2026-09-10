"""Utility script to generate realistic synthetic test images representing handicraft scenarios."""

import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw


def generate_sample_images(output_dir: Path):
    """Generates synthetic test fixtures for various smartphone capture conditions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Normal pottery handicraft: Earthen terracotta pot with geometric folk motifs
    img_normal = np.full((1200, 1200, 3), (220, 215, 210), dtype=np.uint8)  # Neutral floor/table
    # Draw a wooden table grain
    for y in range(0, 1200, 40):
        cv2.line(img_normal, (0, y), (1200, y), (200, 195, 190), 1)

    # Terracotta pot body (warm brownish orange: BGR (30, 80, 180))
    cv2.ellipse(img_normal, (600, 700), (320, 380), 0, 0, 360, (35, 85, 190), -1)
    cv2.ellipse(img_normal, (600, 360), (160, 80), 0, 0, 360, (40, 95, 210), -1)
    # Pot neck and rim
    pts = np.array([[460, 360], [740, 360], [700, 480], [500, 480]], np.int32)
    cv2.fillPoly(img_normal, [pts], (35, 85, 190))
    # Traditional hand-painted motifs (white & ochre)
    cv2.circle(img_normal, (600, 680), 120, (240, 240, 245), 6)
    cv2.circle(img_normal, (600, 680), 60, (50, 170, 220), -1)
    for angle in range(0, 360, 45):
        rad = np.deg2rad(angle)
        cx = int(600 + 120 * np.cos(rad))
        cy = int(680 + 120 * np.sin(rad))
        cv2.circle(img_normal, (cx, cy), 15, (255, 255, 255), -1)

    cv2.imwrite(str(output_dir / "pottery_normal.jpg"), img_normal, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # 2. Blurry handicraft photo (Motion blur)
    kernel_size = 35
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    img_blurry = cv2.filter2D(img_normal, -1, kernel)
    cv2.imwrite(str(output_dir / "pottery_blurry.jpg"), img_blurry, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # 3. Dark/underexposed textile photo (Dark room, low lighting)
    img_dark = (img_normal * 0.15).astype(np.uint8)
    cv2.imwrite(str(output_dir / "textile_dark.jpg"), img_dark, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # 4. Overexposed / harsh glare photo
    img_bright = np.clip(img_normal.astype(np.int32) * 1.8 + 80, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / "handicraft_bright.jpg"), img_bright, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # 5. Product occupies tiny corner (Bad framing / composition)
    img_small = np.full((1200, 1200, 3), (180, 180, 180), dtype=np.uint8)
    # Tiny 100x100 pot in bottom corner
    cv2.ellipse(img_small, (150, 1050), (40, 50), 0, 0, 360, (35, 85, 190), -1)
    cv2.imwrite(str(output_dir / "craft_small.jpg"), img_small, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # 6. Tiny resolution image (< 64px)
    tiny = np.full((32, 32, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(output_dir / "tiny_res.jpg"), tiny)

    # 7. Corrupted file (invalid header/data)
    with open(output_dir / "corrupted.jpg", "wb") as f:
        f.write(b"NOT_A_VALID_IMAGE_FILE_DATA_CORRUPT")

    print(f"Generated sample test images in {output_dir}")


if __name__ == "__main__":
    generate_sample_images(Path(__file__).parent / "images")
