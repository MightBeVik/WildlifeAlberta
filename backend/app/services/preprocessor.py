"""
Image preprocessing utilities for the wildlife detection pipeline.
Handles resizing, normalization, augmentation, and batch processing.
"""
import io
import uuid
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.config import MAX_IMAGE_SIZE, SUPPORTED_EXTENSIONS


def generate_image_id() -> str:
    """Generate a unique image ID."""
    return str(uuid.uuid4())[:8]


def validate_image(filepath: Path) -> bool:
    """Check if a file is a valid supported image."""
    if not filepath.exists():
        return False
    if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


def load_image(filepath: Path) -> Image.Image:
    """Load an image from disk and convert to RGB."""
    img = Image.open(filepath)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def resize_image(img: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    """Resize image while maintaining aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)


def normalize_image(img: Image.Image) -> np.ndarray:
    """Convert image to normalized numpy array (0-1 range)."""
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def enhance_low_light(img: Image.Image, factor: float = 1.5) -> Image.Image:
    """Enhance brightness for low-light camera trap images."""
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(factor)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    return img


def crop_detection(
    img: Image.Image,
    bbox: Tuple[float, float, float, float],
    padding: float = 0.1,
) -> Image.Image:
    """
    Crop a detected region from the image with optional padding.
    bbox format: (x1, y1, x2, y2) in normalized coordinates [0, 1].
    """
    w, h = img.size
    x1, y1, x2, y2 = bbox

    # Convert normalized coords to pixel coords
    px1 = int(x1 * w)
    py1 = int(y1 * h)
    px2 = int(x2 * w)
    py2 = int(y2 * h)

    # Add padding
    pad_w = int((px2 - px1) * padding)
    pad_h = int((py2 - py1) * padding)

    px1 = max(0, px1 - pad_w)
    py1 = max(0, py1 - pad_h)
    px2 = min(w, px2 + pad_w)
    py2 = min(h, py2 + pad_h)

    return img.crop((px1, py1, px2, py2))


def get_image_info(filepath: Path) -> dict:
    """Get basic image metadata."""
    try:
        with Image.open(filepath) as img:
            return {
                "width": img.size[0],
                "height": img.size[1],
                "format": img.format,
                "mode": img.mode,
                "size_bytes": filepath.stat().st_size,
            }
    except Exception as e:
        return {"error": str(e)}


def augment_image(img: Image.Image, augmentation: str) -> Image.Image:
    """Apply a single augmentation to an image."""
    if augmentation == "flip_horizontal":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif augmentation == "flip_vertical":
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    elif augmentation == "rotate_90":
        return img.rotate(90, expand=True)
    elif augmentation == "brightness":
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(np.random.uniform(0.7, 1.3))
    elif augmentation == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=1))
    else:
        return img


def image_to_bytes(img: Image.Image, format: str = "JPEG") -> bytes:
    """Convert a PIL Image to bytes."""
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()
