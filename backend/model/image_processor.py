"""
Remote Sensing Image Preprocessing & Format Adapter
Author: Sat_Assist Team

Supports:
- Standard formats: JPG, JPEG, PNG, WebP
- Remote Sensing TIFF / GeoTIFF (8-bit, 16-bit, multi-spectral bands)
- Dynamic contrast stretching (2% - 98% percentile normalization)
- Aspect-ratio preserving smart resizing for VLM ingestion
- Thumbnail generation for frontend preview
"""

import io
import os
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from typing import Tuple, Optional

try:
    import tifffile
except ImportError:
    tifffile = None

class SatelliteImageProcessor:
    MAX_VLM_DIMENSION = 1024  # Max width/height for VLM inference to optimize VRAM
    THUMBNAIL_SIZE = (256, 256)

    @classmethod
    def load_image(cls, image_source) -> Image.Image:
        """
        Loads an image from a file path, bytes, or file-like object.
        Automatically handles TIFF, 16-bit depths, and multi-channel remote sensing data.
        """
        if isinstance(image_source, (str, Path)):
            path = Path(image_source)
            if not path.exists():
                raise FileNotFoundError(f"Image not found at: {path}")
            
            ext = path.suffix.lower()
            if ext in [".tif", ".tiff"] and tifffile is not None:
                return cls._load_tiff_from_file(str(path))
            else:
                img = Image.open(path)
                return cls._standardize_pil_image(img)

        elif isinstance(image_source, bytes):
            # Try PIL first
            try:
                img = Image.open(io.BytesIO(image_source))
                return cls._standardize_pil_image(img)
            except Exception:
                if tifffile is not None:
                    arr = tifffile.imread(io.BytesIO(image_source))
                    return cls._normalize_numpy_array(arr)
                raise ValueError("Could not parse image bytes.")

        elif hasattr(image_source, "read"):
            # File-like object
            content = image_source.read()
            return cls.load_image(content)

        elif isinstance(image_source, Image.Image):
            return cls._standardize_pil_image(image_source)

        else:
            raise TypeError(f"Unsupported image source type: {type(image_source)}")

    @classmethod
    def _standardize_pil_image(cls, img: Image.Image) -> Image.Image:
        """Auto-rotates EXIF orientation, converts palette/grayscale/RGBA to RGB."""
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    @classmethod
    def _load_tiff_from_file(cls, path: str) -> Image.Image:
        """Loads and normalizes TIFF / GeoTIFF array to 8-bit RGB."""
        arr = tifffile.imread(path)
        return cls._normalize_numpy_array(arr)

    @classmethod
    def _normalize_numpy_array(cls, arr: np.ndarray) -> Image.Image:
        """
        Normalizes multi-spectral / 16-bit satellite arrays to standard 8-bit RGB.
        Uses 2% - 98% percentile contrast stretching.
        """
        if arr.ndim == 2:
            # Grayscale: repeat to 3 channels
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3:
            # Check channel ordering: (C, H, W) vs (H, W, C)
            if arr.shape[0] in [1, 3, 4] and arr.shape[0] < arr.shape[1]:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[-1] > 3:
                # Take first 3 channels (e.g. RGB from multispectral)
                arr = arr[:, :, :3]
            elif arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)

        # Percentile contrast stretch to 8-bit [0, 255]
        p2, p98 = np.percentile(arr, (2, 98))
        if p98 > p2:
            arr_stretched = np.clip((arr - p2) / (p98 - p2), 0, 1) * 255.0
        else:
            arr_stretched = np.clip(arr, 0, 255)

        arr_uint8 = arr_stretched.astype(np.uint8)
        return Image.fromarray(arr_uint8, mode="RGB")

    @classmethod
    def prepare_for_vlm(cls, img: Image.Image, max_dim: int = MAX_VLM_DIMENSION) -> Image.Image:
        """
        Resizes image while preserving aspect ratio so it fits within max_dim.
        Prevents VRAM overflow while retaining fine remote sensing details.
        """
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return img

    @classmethod
    def generate_thumbnail_bytes(cls, img: Image.Image, size: Tuple[int, int] = THUMBNAIL_SIZE) -> bytes:
        """Generates a compressed JPEG thumbnail for fast web delivery."""
        thumb = img.copy()
        thumb.thumbnail(size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    @classmethod
    def get_image_metadata(cls, img: Image.Image) -> dict:
        """Returns standard metadata for frontend display."""
        w, h = img.size
        return {
            "width": w,
            "height": h,
            "aspect_ratio": round(w / h, 3) if h > 0 else 1.0,
            "mode": img.mode,
            "format": "RGB"
        }
