from enum import Enum
from typing import List, Optional, Dict, Any

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Graceful fallback when running in minimal environments without pydantic installed
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self) -> Dict[str, Any]:
            res = {}
            for k, v in self.__dict__.items():
                if hasattr(v, "model_dump"):
                    res[k] = v.model_dump()
                elif isinstance(v, Enum):
                    res[k] = v.value
                else:
                    res[k] = v
            return res

    def Field(default=None, default_factory=None, **kwargs):
        if default_factory is not None:
            return default_factory()
        return default


class ExtractionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    MANUAL_ENTRY = "MANUAL_ENTRY"


class QualityStatus(str, Enum):
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"
    UNUSABLE = "UNUSABLE"
    UNKNOWN = "UNKNOWN"


class BoundingBox(BaseModel):
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    raw_points: Optional[List[List[float]]] = None


class OCRToken(BaseModel):
    text: str = ""
    confidence: float = 0.0
    bounding_box: BoundingBox = Field(default_factory=BoundingBox)
    line_index: Optional[int] = None


class ImageQuality(BaseModel):
    image_quality_score: float = Field(default=0.0)
    quality_status: QualityStatus = QualityStatus.UNKNOWN
    quality_issues: List[str] = Field(default_factory=list)


class EvidenceField(BaseModel):
    value: Any = None
    confidence: float = 0.0
    source_text: Optional[str] = None
    evidence: Optional[str] = None


class ValidationMetadata(BaseModel):
    timezone: str = "Africa/Khartoum"
    current_date_in_khartoum: str = ""
    is_future_date: bool = False
    is_today_date: bool = False
    date_interpretation_notes: str = ""


class StructuredReceiptData(BaseModel):
    amount: Optional[float] = None
    formatted_amount: Optional[str] = None
    currency: str = "SDG"
    sender_name: Optional[str] = None
    sender_phone: Optional[str] = None
    sender_account: Optional[str] = None
    receiver_name: Optional[str] = None
    receiver_phone: Optional[str] = None
    receiver_account: Optional[str] = None
    transaction_date: Optional[str] = None  # YYYY-MM-DD
    transaction_time: Optional[str] = None  # HH:mm:ss
    reference_number: Optional[str] = None
    transaction_id: Optional[str] = None
    bank_name: Optional[str] = None
    wallet_name: Optional[str] = None
    narration: Optional[str] = None
    transaction_type: Optional[str] = None
    transaction_status: Optional[str] = None
    merchant_name: Optional[str] = None
    raw_text: str = ""
    image_quality: ImageQuality = Field(default_factory=ImageQuality)
    field_confidences: Dict[str, float] = Field(default_factory=dict)
    overall_confidence: float = 0.0
    warnings: List[str] = Field(default_factory=list)
    review_required: bool = True
    extraction_version: str = "b2b-ocr-v1"
    evidence_map: Optional[Dict[str, Any]] = None
    validation_metadata: Optional[ValidationMetadata] = None
    result_status: ExtractionStatus = ExtractionStatus.REVIEW_REQUIRED


class OCRRequest(BaseModel):
    imageBase64: str = ""
    mimeType: str = "image/jpeg"


class OCRResponse(BaseModel):
    success: bool = True
    data: StructuredReceiptData = Field(default_factory=StructuredReceiptData)
