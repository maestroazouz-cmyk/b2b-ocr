import logging
from typing import List, Optional
import numpy as np

from app.models.schemas import OCRToken, BoundingBox

logger = logging.getLogger("b2b_ocr.engine")


class PaddleOCREngine:
    """
    PaddleOCR inference engine with Arabic/English multilingual pipeline.
    Preserves raw tokens, confidence scores, and spatial polygon bounding boxes.
    """

    _instance: Optional["PaddleOCREngine"] = None
    _ocr = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PaddleOCREngine, cls).__new__(cls)
            cls._instance._initialize_ocr()
        return cls._instance

    def _initialize_ocr(self):
        try:
            from paddleocr import PaddleOCR
            logger.info("Initializing PaddleOCR with Arabic language & text angle classifier (CPU runtime)...")
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="ar",  # Supports Arabic, Latin, and digits seamlessly
                show_log=False,
                use_gpu=False,
            )
            logger.info("PaddleOCR engine initialized successfully.")
        except ImportError as e:
            logger.warning(
                f"PaddleOCR package not installed in current environment ({str(e)}). Engine ready for fallback mode."
            )
            self._ocr = None
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            self._ocr = None

    def is_available(self) -> bool:
        return self._ocr is not None

    def extract_tokens(self, image_bgr: np.ndarray) -> List[OCRToken]:
        """
        Runs PaddleOCR inference on image array and returns vertically sorted OCRToken list.
        """
        if self._ocr is None:
            # Re-attempt initialization
            self._initialize_ocr()
            if self._ocr is None:
                raise RuntimeError(
                    "PaddleOCR engine is unavailable. Please install paddleocr and paddlepaddle dependencies."
                )

        try:
            # PaddleOCR returns: [ [ [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ], ( "text", confidence ) ], ... ]
            results = self._ocr.ocr(image_bgr, cls=True)
        except Exception as e:
            logger.error(f"PaddleOCR inference exception: {str(e)}")
            raise RuntimeError(f"OCR inference failed: {str(e)}")

        tokens: List[OCRToken] = []

        if not results or not results[0]:
            return tokens

        raw_boxes = results[0]

        for item in raw_boxes:
            if not item or len(item) < 2:
                continue

            points = item[0]  # 4 vertices
            text_conf = item[1]  # (text, confidence)

            if not text_conf or len(text_conf) < 2:
                continue

            text = str(text_conf[0]).strip()
            confidence = float(text_conf[1]) if text_conf[1] is not None else 0.0

            if not text:
                continue

            # Compute axis-aligned bounding box from 4 polygon points
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)

            bbox = BoundingBox(
                x=float(min_x),
                y=float(min_y),
                width=float(max_x - min_x),
                height=float(max_y - min_y),
                raw_points=points,
            )

            tokens.append(
                OCRToken(
                    text=text,
                    confidence=confidence,
                    bounding_box=bbox,
                )
            )

        # Sort tokens primarily top-to-bottom (Y), then left-to-right (X)
        tokens.sort(key=lambda t: (t.bounding_box.y, t.bounding_box.x))

        # Assign line index based on vertical proximity
        if tokens:
            current_line = 0
            tokens[0].line_index = 0
            for i in range(1, len(tokens)):
                prev = tokens[i - 1]
                curr = tokens[i]
                # If vertical center is within 0.6 * height of previous token, consider same line
                if abs(curr.bounding_box.y - prev.bounding_box.y) <= (max(curr.bounding_box.height, 15) * 0.6):
                    curr.line_index = current_line
                else:
                    current_line += 1
                    curr.line_index = current_line

        return tokens


ocr_engine = PaddleOCREngine()
