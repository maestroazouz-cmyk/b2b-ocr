import base64
import io
import logging
from typing import Tuple, List
import numpy as np
from PIL import Image, ImageOps

from app.models.schemas import ImageQuality, QualityStatus

logger = logging.getLogger("b2b_ocr.preprocessing")


class ImagePreprocessor:
    """
    Robust image preprocessing pipeline for paper/mobile receipt vouchers:
    - Base64 sanitization & validation
    - Dimension normalization (max 2048px for OCR efficiency)
    - EXIF orientation correction
    - Contrast/Illumination assessment
    - Laplacian variance blur detection
    """

    MAX_DIMENSION = 2048
    MIN_DIMENSION = 150

    @classmethod
    def decode_base64_image(cls, image_b64: str) -> Tuple[Image.Image, str]:
        """
        Decodes Base64 string into PIL Image and detects MIME type.
        """
        if not image_b64 or not isinstance(image_b64, str):
            raise ValueError("Empty or invalid imageBase64 payload.")

        mime_type = "image/jpeg"
        clean_b64 = image_b64.strip()

        # Strip Data URI scheme if present
        if clean_b64.startswith("data:"):
            header_end = clean_b64.find(",")
            if header_end != -1:
                header = clean_b64[:header_end]
                if "image/png" in header:
                    mime_type = "image/png"
                elif "image/webp" in header:
                    mime_type = "image/webp"
                elif "image/jpeg" in header or "image/jpg" in header:
                    mime_type = "image/jpeg"
                clean_b64 = clean_b64[header_end + 1 :]

        try:
            image_bytes = base64.b64decode(clean_b64)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 byte stream: {str(e)}")

        if len(image_bytes) < 100:
            raise ValueError("Decoded image payload is too small (< 100 bytes).")

        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image.load()
        except Exception as e:
            raise ValueError(f"Unsupported or corrupt image structure: {str(e)}")

        # Validate MIME format
        format_lower = (pil_image.format or "").lower()
        if format_lower not in ["jpeg", "jpg", "png", "webp", "bmp", "tiff"]:
            raise ValueError(f"Unsupported image format: {pil_image.format}. Only standard image types supported.")

        return pil_image, mime_type

    @classmethod
    def assess_quality(cls, pil_image: Image.Image) -> ImageQuality:
        """
        Evaluates image clarity, resolution, and exposure defects.
        """
        issues: List[str] = []
        width, height = pil_image.size

        # 1. Resolution Check
        if width < cls.MIN_DIMENSION or height < cls.MIN_DIMENSION:
            issues.append("VERY_LOW_RESOLUTION")
        elif width < 400 or height < 400:
            issues.append("LOW_RESOLUTION")

        # Convert to Grayscale for numerical analysis
        gray_img = pil_image.convert("L")
        gray_arr = np.array(gray_img, dtype=np.uint8)

        # 2. Exposure Check (Mean brightness & Dynamic range)
        mean_brightness = float(np.mean(gray_arr))
        std_brightness = float(np.std(gray_arr))

        if mean_brightness < 40.0:
            issues.append("UNDEREXPOSED_DARK")
        elif mean_brightness > 230.0:
            issues.append("OVEREXPOSED_WASHED_OUT")

        if std_brightness < 20.0:
            issues.append("LOW_CONTRAST")

        # 3. Blur Detection (Laplacian Variance approximation)
        # Using 3x3 kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        h, w = gray_arr.shape
        if h > 10 and w > 10:
            center = gray_arr[1:-1, 1:-1].astype(np.float32)
            top = gray_arr[0:-2, 1:-1].astype(np.float32)
            bottom = gray_arr[2:, 1:-1].astype(np.float32)
            left = gray_arr[1:-1, 0:-2].astype(np.float32)
            right = gray_arr[1:-1, 2:].astype(np.float32)

            laplacian = (top + bottom + left + right) - (4.0 * center)
            laplacian_var = float(np.var(laplacian))

            if laplacian_var < 50.0:
                issues.append("HIGH_BLUR")
            elif laplacian_var < 100.0:
                issues.append("MODERATE_BLUR")
        else:
            laplacian_var = 0.0

        # Quality Score Calculation
        score = 1.0
        if "VERY_LOW_RESOLUTION" in issues:
            score -= 0.4
        elif "LOW_RESOLUTION" in issues:
            score -= 0.15

        if "HIGH_BLUR" in issues:
            score -= 0.35
        elif "MODERATE_BLUR" in issues:
            score -= 0.15

        if "UNDEREXPOSED_DARK" in issues or "OVEREXPOSED_WASHED_OUT" in issues:
            score -= 0.2

        if "LOW_CONTRAST" in issues:
            score -= 0.15

        score = max(0.0, min(1.0, score))

        if score >= 0.85:
            status = QualityStatus.GOOD
        elif score >= 0.65:
            status = QualityStatus.ACCEPTABLE
        elif score >= 0.40:
            status = QualityStatus.POOR
        else:
            status = QualityStatus.UNUSABLE

        return ImageQuality(
            image_quality_score=round(score, 2),
            quality_status=status,
            quality_issues=issues,
        )

    @classmethod
    def preprocess_image(cls, pil_image: Image.Image) -> Tuple[np.ndarray, ImageQuality]:
        """
        Normalizes orientation, scales if oversized, converts to RGB BGR numpy array.
        """
        # Fix EXIF rotation
        try:
            pil_image = ImageOps.exif_transpose(pil_image)
        except Exception:
            pass

        # Convert palette/RGBA to RGB
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Assess quality before downscaling
        quality = cls.assess_quality(pil_image)

        # Scale down if exceeds max bounds
        w, h = pil_image.size
        if max(w, h) > cls.MAX_DIMENSION:
            scale = cls.MAX_DIMENSION / float(max(w, h))
            new_w = int(w * scale)
            new_h = int(h * scale)
            pil_image = pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Convert to BGR array for PaddleOCR / OpenCV
        rgb_arr = np.array(pil_image, dtype=np.uint8)
        bgr_arr = rgb_arr[:, :, ::-1].copy()

        return bgr_arr, quality
