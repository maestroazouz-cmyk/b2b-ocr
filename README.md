# B2B Sudanese Payment Receipt OCR Microservice (`b2b-ocr`)

An enterprise-grade, standalone Optical Character Recognition (OCR) and financial voucher parser service built with **PaddleOCR (PP-OCRv4 Multilingual Arabic & Latin)** and **FastAPI**.

This microservice provides high-precision, evidence-grounded document extraction for Sudanese banking vouchers, corporate payment receipts, and mobile wallets (Bank of Khartoum Bankak, Faisal Islamic Bank Fawry, Omdurman National Bank ONB, etc.) without relying on external LLM APIs (Gemini/OpenAI).

---

## Key Features

- **Multilingual OCR Engine**: Powered by PaddlePaddle CPU runtime with OpenMP/MKLDNN acceleration.
- **Sudanese Banking Domain Heuristics**: Deterministic parsing for Sudanese Arabic numerals (٠١٢٣٤٥٦٧٨٩), localized date/time formats, phone number normalization (`09XXXXXXXX` / `01XXXXXXXX`), and transaction reference numbers.
- **Anti-Hallucination Constitution**: Strict visual evidence grounding. If a field is not visibly present on the receipt voucher, the parser returns `null` with `0.0` confidence instead of guessing or synthesizing data.
- **Temporal Africa/Khartoum Timezone Anchoring**: Evaluates execution dates against Khartoum local time (`UTC+2`), flagging invalid future transaction dates without mutating source strings.
- **Standardized B2B Contract**: 100% drop-in compatible with the Zain B2B Corporate Banking & Collections ecosystem.
- **Cloud Run Ready**: Containerized with Debian Bullseye, pre-cached model weights, dynamic `$PORT` binding, and health monitoring.

---

## Project Structure

```
b2b-ocr/
├── Dockerfile                  # Production container definition (Debian Bullseye, Python 3.10)
├── requirements.txt            # Pinned dependencies (PaddleOCR, PaddlePaddle, FastAPI)
├── main.py                     # FastAPI server & route handlers
├── README.md                   # System documentation
├── DEPLOYMENT.md               # Google Cloud Run deployment manual
├── .gitignore                  # Git hygiene rules
├── .env.example                # Environment variables template
│
├── app/
│   ├── models/
│   │   └── schemas.py          # Pydantic schemas & B2B contract definitions
│   ├── ocr/
│   │   ├── engine.py           # PaddleOCR singleton wrapper & token polygon extractor
│   │   └── preprocessing.py    # Laplacian blur check, EXIF transpose, exposure rating
│   ├── parser/
│   │   ├── receipt_parser.py   # Master coordinator & evidence map generator
│   │   ├── transaction_parser.py # Ref # / Txn ID extractor with exclusion guardrails
│   │   ├── amount_parser.py    # Decimal amount & currency detector
│   │   ├── phone_parser.py     # Sudanese mobile normalizer (09/01)
│   │   ├── date_parser.py      # Timezone-aware date & time extractor
│   │   ├── party_parser.py     # Sender / Beneficiary name extractor
│   │   └── bank_parser.py      # Evidence-based bank & wallet dictionary
│   ├── validation/
│   │   └── validator.py        # Business rule auditor & warning engine
│   └── confidence/
│       └── scorer.py           # Weighted evidence-based confidence scorer
│
└── tests/
    └── test_parser.py          # Automated unit tests for Sudanese voucher heuristics
```

---

## API Endpoints

### 1. Health Probe (`GET /health`)
Returns service status, operating timezone, and engine initialization state.

**Response:**
```json
{
  "status": "ok",
  "service": "b2b-paddleocr-service",
  "ocr_engine": "PaddleOCR PP-OCRv4 (Arabic/English)",
  "ocr_engine_ready": true,
  "parser_version": "b2b-ocr-v1",
  "timezone": "Africa/Khartoum (UTC+2)"
}
```

---

### 2. Receipt OCR Extraction (`POST /ocr`)

**Request Payload:**
```json
{
  "imageBase64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "mimeType": "image/jpeg"
}
```

**Supported Image Types:** `image/jpeg`, `image/png`, `image/webp`.

**Success Response (`HTTP 200`):**
```json
{
  "success": true,
  "data": {
    "amount": 150000.0,
    "formatted_amount": "150,000.00 SDG",
    "currency": "SDG",
    "sender_name": "شركة النيل للتوريدات",
    "sender_phone": "0912345678",
    "sender_account": "18293041",
    "receiver_name": "شركة زين للاتصالات",
    "receiver_phone": null,
    "receiver_account": "99201482",
    "transaction_date": "2026-09-15",
    "transaction_time": "14:30:00",
    "reference_number": "BOK-9842104",
    "transaction_id": "BOK-9842104",
    "bank_name": "Bank of Khartoum",
    "wallet_name": "Bankak",
    "narration": "سداد فاتورة رقم 4021",
    "transaction_type": "BANK_TRANSFER",
    "transaction_status": "SUCCESS",
    "merchant_name": "Zain Telecom B2B",
    "raw_text": "بنك الخرطوم\nإشعار تحويل مالي\n...",
    "image_quality": {
      "image_quality_score": 0.92,
      "quality_status": "GOOD",
      "quality_issues": []
    },
    "field_confidences": {
      "amount": 0.96,
      "reference_number": 0.98,
      "sender_phone": 0.95,
      "transaction_date": 0.94,
      "bank_name": 0.96,
      "sender_name": 0.92
    },
    "overall_confidence": 0.95,
    "warnings": [],
    "review_required": false,
    "extraction_version": "b2b-ocr-v1",
    "result_status": "SUCCESS"
  }
}
```

---

## Local Development & Testing

### 1. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Parser Unit Tests
```bash
python3 tests/test_parser.py
```

### 3. Start Local Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Production Container Execution

```bash
docker build -t b2b-ocr-service .
docker run -p 8000:8000 --cpus=2 --memory=2g b2b-ocr-service
```
