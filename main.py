import logging
import time
import sys
import os
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.models.schemas import (
    OCRRequest,
    OCRResponse,
    StructuredReceiptData,
    ExtractionStatus,
)
from app.ocr.preprocessing import ImagePreprocessor
from app.ocr.engine import ocr_engine
from app.parser.receipt_parser import SudaneseReceiptParser
from app.validation.validator import ReceiptValidator

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("b2b_ocr.main")

app = FastAPI(
    title="B2B Sudanese Payment Receipt OCR Microservice",
    description="Dedicated PaddleOCR engine & deterministic Sudanese payment voucher parser for B2B Corporate Banking",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check probe returning OCR engine readiness and operating timezone.
    Strictly reports ocr_engine_ready: true only if PaddleOCR is initialized.
    """
    is_ready = ocr_engine.is_available()
    return {
        "status": "ok",
        "service": "b2b-paddleocr-service",
        "ocr_engine": "PaddleOCR PP-OCRv4 (Arabic/English)",
        "ocr_engine_ready": is_ready,
        "parser_version": "b2b-ocr-v1",
        "timezone": "Africa/Khartoum (UTC+2)",
    }


@app.post("/ocr", response_model=OCRResponse, status_code=status.HTTP_200_OK)
async def process_ocr(request: OCRRequest):
    """
    Primary OCR extraction endpoint conforming to B2B frontend contract.
    Accepts Base64 image payload -> Preprocesses -> Extracts Tokens -> Parses Sudanese Entities.
    """
    start_time = time.time()
    logger.info(f"Incoming OCR request: MIME={request.mimeType} | PayloadLength={len(request.imageBase64)}")

    # 1. Validation & Preprocessing
    try:
        raw_image, detected_mime = ImagePreprocessor.decode_base64_image(request.imageBase64)
        processed_image, quality_assessment = ImagePreprocessor.preprocess_image(raw_image)
    except ValueError as val_err:
        logger.warning(f"Image preprocessing validation error: {str(val_err)}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "errorCode": "INVALID_IMAGE_PAYLOAD",
                "error": str(val_err),
            },
        )
    except Exception as err:
        logger.error(f"Unexpected image decoding failure: {str(err)}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "errorCode": "IMAGE_DECODING_ERROR",
                "error": "Failed to decode submitted receipt image.",
            },
        )

    preprocess_duration = round((time.time() - start_time) * 1000, 1)

    # 2. Token Extraction via PaddleOCR
    ocr_start = time.time()
    try:
        tokens = ocr_engine.extract_tokens(processed_image)
    except Exception as err:
        logger.error(f"PaddleOCR token extraction error: {str(err)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "errorCode": "OCR_ENGINE_ERROR",
                "error": "Internal OCR inference engine encountered an error.",
                "technicalDetails": str(err),
            },
        )

    ocr_duration = round((time.time() - ocr_start) * 1000, 1)

    # 3. Deterministic Sudanese Receipt Parsing
    parser_start = time.time()
    structured_data = SudaneseReceiptParser.parse(tokens, quality_assessment)

    # 4. Validation Engine Check
    is_valid, validation_warnings = ReceiptValidator.validate(structured_data)
    if validation_warnings:
        structured_data.warnings.extend(validation_warnings)

    parser_duration = round((time.time() - parser_start) * 1000, 1)
    total_duration = round((time.time() - start_time) * 1000, 1)

    logger.info(
        f"OCR Success in {total_duration}ms (Preprocess: {preprocess_duration}ms, "
        f"OCR: {ocr_duration}ms, Parser: {parser_duration}ms) | "
        f"Status={structured_data.result_status} | "
        f"Phone={structured_data.sender_phone} | "
        f"Ref={structured_data.reference_number} | "
        f"Amount={structured_data.formatted_amount} | "
        f"Confidence={structured_data.overall_confidence}"
    )

    return OCRResponse(
        success=True,
        data=structured_data,
    )


@app.post("/test-file")
async def test_file_upload(file: UploadFile = File(...)):
    """
    Direct multipart file upload test endpoint.
    """
    import base64

    contents = await file.read()
    b64_str = base64.b64encode(contents).decode("utf-8")
    req = OCRRequest(imageBase64=b64_str, mimeType=file.content_type or "image/jpeg")
    return await process_ocr(req)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
