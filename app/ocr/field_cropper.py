import logging
from typing import Optional, List, Tuple
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.models.schemas import BoundingBox, OCRToken

logger = logging.getLogger("b2b_ocr.field_cropper")


class FieldCropper:
    """
    Dedicated second-stage image cropper and preprocessor for numeric / structured receipt regions.
    Applies high-resolution upscaling, grayscale contrast enhancement, and noise reduction.
    """

    @classmethod
    def crop_region(
        cls,
        image_bgr: np.ndarray,
        bbox: BoundingBox,
        pad_x: int = 8,
        pad_y: int = 6,
    ) -> Optional[np.ndarray]:
        """
        Crops a region from the full BGR image based on bounding box with padding.
        """
        if image_bgr is None or image_bgr.size == 0:
            return None

        h, w = image_bgr.shape[:2]
        x1 = max(0, int(bbox.x - pad_x))
        y1 = max(0, int(bbox.y - pad_y))
        x2 = min(w, int(bbox.x + bbox.width + pad_x))
        y2 = min(h, int(bbox.y + bbox.height + pad_y))

        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        crop = image_bgr[y1:y2, x1:x2].copy()
        return crop

    @classmethod
    def enhance_numeric_crop(cls, crop_bgr: np.ndarray, upscale_factor: float = 2.5) -> np.ndarray:
        """
        Optimizes a cropped numeric/text region for high-precision character recognition:
        - Converts to PIL Image
        - Bicubic 2.5x upscale
        - Contrast sharpening
        - Preserves decimal dots, colons, hyphens, and digit geometry
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr

        try:
            # Convert BGR to RGB PIL
            rgb_arr = crop_bgr[:, :, ::-1]
            pil_img = Image.fromarray(rgb_arr)

            # Upscale
            orig_w, orig_h = pil_img.size
            new_w = max(50, int(orig_w * upscale_factor))
            new_h = max(20, int(orig_h * upscale_factor))
            upscaled = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)

            # Convert to Grayscale & Contrast Enhancement
            gray = upscaled.convert("L")
            enhancer = ImageEnhance.Contrast(gray)
            enhanced = enhancer.enhance(1.8)

            # Convert back to BGR numpy array
            enhanced_arr = np.array(enhanced, dtype=np.uint8)
            # Create 3-channel BGR from grayscale for PaddleOCR compatibility
            bgr_enhanced = np.stack([enhanced_arr] * 3, axis=-1)
            return bgr_enhanced
        except Exception as e:
            logger.warning(f"Error enhancing numeric crop: {e}")
            return crop_bgr
