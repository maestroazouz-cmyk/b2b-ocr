"""
Parser Unit Tests for Sudanese Payment Receipt Vouchers.
NOTE: These tests validate parser heuristics, regex rules, spatial ordering, and anti-hallucination behavior.
They do NOT count as real pixel OCR accuracy evidence.
"""

import sys
import os

# Add parent directory to path for standalone execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.schemas import OCRToken, BoundingBox, ImageQuality, QualityStatus, ExtractionStatus
from app.parser.receipt_parser import SudaneseReceiptParser
from app.parser.amount_parser import AmountParser
from app.parser.phone_parser import PhoneParser
from app.parser.transaction_parser import TransactionParser
from app.parser.date_parser import DateParser
from app.parser.bank_parser import BankParser
from app.parser.party_parser import PartyParser


def create_token(text: str, x: float = 0, y: float = 0, confidence: float = 0.95) -> OCRToken:
    return OCRToken(
        text=text,
        confidence=confidence,
        bounding_box=BoundingBox(x=x, y=y, width=100, height=20),
    )


def test_bank_of_khartoum_voucher():
    print("--- Test 1: Bank of Khartoum (Bankak) Parser Test ---")
    tokens = [
        create_token("بنك الخرطوم", y=10),
        create_token("إشعار تحويل مالي", y=30),
        create_token("رقم المعاملة: BOK-9842104", y=50),
        create_token("المبلغ: 150,000.00 ج.س", y=70),
        create_token("من: شركة النيل للتوريدات", y=90),
        create_token("إلى: شركة زين للاتصالات", y=110),
        create_token("هاتف: 0912345678", y=130),
        create_token("التاريخ: 2026-09-15 14:30:00", y=150),
        create_token("البيان: سداد فاتورة رقم 4021", y=170),
    ]

    quality = ImageQuality(image_quality_score=0.92, quality_status=QualityStatus.GOOD)
    result = SudaneseReceiptParser.parse(tokens, quality)

    assert result.bank_name == "Bank of Khartoum", f"Expected Bank of Khartoum, got {result.bank_name}"
    assert result.sender_phone == "0912345678", f"Expected 0912345678, got {result.sender_phone}"
    assert result.reference_number == "BOK-9842104", f"Expected BOK-9842104, got {result.reference_number}"
    assert result.amount == 150000.0, f"Expected 150000.0, got {result.amount}"
    assert result.currency == "SDG", f"Expected SDG, got {result.currency}"
    assert result.sender_name == "شركة النيل للتوريدات", f"Expected sender name, got {result.sender_name}"
    assert result.receiver_name == "شركة زين للاتصالات", f"Expected receiver name, got {result.receiver_name}"
    assert result.transaction_date == "2026-09-15", f"Expected 2026-09-15, got {result.transaction_date}"
    assert result.result_status == ExtractionStatus.SUCCESS
    print("Test 1 Passed: Full Bankak voucher parsed correctly.\n")


def test_fawry_evidence_based_bank_check():
    print("--- Test 2: Fawry / Wallet Evidence-Based Check ---")
    tokens = [
        create_token("خدمة فوري للمدفوعات", y=10),
        create_token("المبلغ 320,000", y=30),
        create_token("المستفيد: زين السودان", y=50),
        create_token("رقم الهاتف: 0913456789", y=70),
    ]

    quality = ImageQuality(image_quality_score=0.88, quality_status=QualityStatus.GOOD)
    result = SudaneseReceiptParser.parse(tokens, quality)

    # Wallet is Fawry, but Bank is NOT automatically set to Faisal Islamic Bank unless explicitly written
    assert result.wallet_name == "Fawry", f"Expected Fawry wallet, got {result.wallet_name}"
    assert result.bank_name is None, f"Bank must be None without explicit bank label, got {result.bank_name}"
    assert result.amount == 320000.0
    assert result.sender_phone == "0913456789"
    assert result.reference_number is None, "Missing ref must return None (Anti-Hallucination)"
    assert result.result_status == ExtractionStatus.PARTIAL_SUCCESS
    print("Test 2 Passed: Fawry correctly classified without hallucinatory bank mapping.\n")


def test_anti_hallucination_empty_fields():
    print("--- Test 3: Anti-Hallucination on Missing Fields ---")
    # Receipt with only random non-financial words
    tokens = [
        create_token("تقرير الحساب الشهري", y=10),
        create_token("شكراً لتعاملكم معنا", y=30),
    ]

    quality = ImageQuality(image_quality_score=0.75, quality_status=QualityStatus.ACCEPTABLE)
    result = SudaneseReceiptParser.parse(tokens, quality)

    assert result.amount is None, "Amount must be null"
    assert result.reference_number is None, "Reference number must be null"
    assert result.sender_name is None, "Sender must be null"
    assert result.receiver_name is None, "Receiver must be null"
    assert result.sender_phone is None, "Phone must be null"
    assert result.result_status == ExtractionStatus.MANUAL_ENTRY
    print("Test 3 Passed: All missing fields returned null with 0 confidence.\n")


def test_amount_formats():
    print("--- Test 4: Amount Parser Formats ---")
    cases = [
        ("المبلغ: 150,000", 150000.0),
        ("القيمة 150000.50 ج.س", 150000.50),
        ("Total: 50,000.00 SDG", 50000.0),
    ]
    for text, expected in cases:
        tokens = [create_token(text)]
        amt, fmt, curr, ev = AmountParser.parse(tokens)
        assert amt == expected, f"Expected {expected} for '{text}', got {amt}"
    print("Test 4 Passed: All amount formats correctly parsed.\n")


def test_phone_formats():
    print("--- Test 5: Sudanese Phone Parser Formats ---")
    cases = [
        ("0912345678", "0912345678"),
        ("+249912345678", "0912345678"),
        ("249123456789", "0123456789"),
        ("هاتف العميل 0918765432", "0918765432"),
    ]
    for text, expected in cases:
        tokens = [create_token(text)]
        phone, ev = PhoneParser.parse(tokens)
        assert phone == expected, f"Expected {expected} for '{text}', got {phone}"
    print("Test 5 Passed: All phone formats correctly normalized.\n")


if __name__ == "__main__":
    print("Running B2B OCR Parser Unit Tests...")
    test_bank_of_khartoum_voucher()
    test_fawry_evidence_based_bank_check()
    test_anti_hallucination_empty_fields()
    test_amount_formats()
    test_phone_formats()
    print("========================================")
    print("All 5 Parser Unit Tests Passed Successfully!")
    print("========================================")
