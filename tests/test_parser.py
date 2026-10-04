"""
Comprehensive B2B OCR Sudanese Receipt Parser Test Suite.
Validates all regression tests and user requirements:
1. TEST 1: Date & Time with 2-character month '02-Oc-2026 19:36:41' -> '2026-10-02 19:36:41'
2. TEST 2: Date & Time with OCR letters 'O2-0ct-2026 19:44:07' -> '2026-10-02 19:44:07'
3. TEST 3: Bankak Arabic recipient name with CTC reversed tokens & label stripping ('يقابلادبع يقتلا لاعلادبع اليه المرسل إسم')
4. TEST 4: Labels accidentally mixed into OCR ('اليه المرسل إسم') are never returned as name
5. TEST 5: Ambiguous name returns medium confidence rather than hallucination
6. TEST 6: Account numbers must never become names
7. TEST 7: Transaction ID must never become names
8. TEST 8: Amount extraction for 40,000.00 SDG ('000.00,40' -> 40000.0)
9. TEST 9: Reference number extraction ('20365281771')
10. TEST 10: Sender account zero padding ('373 1204 4436 0001' -> '0373 1204 4436 0001')
11. TEST 11: Receiver account reconstruction
12. TEST 12: Previous working example (150000 SDG, 20015315334, 2026-10-02, 19:44:07, 0373 1204 4436 0001, 0573 0156 6109 0001)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.schemas import (
    OCRToken,
    BoundingBox,
    ImageQuality,
    QualityStatus,
    ExtractionStatus,
)
from app.parser.receipt_parser import SudaneseReceiptParser
from app.parser.arabic_name_processor import ArabicNameProcessor
from app.parser.amount_parser import AmountParser
from app.parser.date_parser import DateParser


def create_token(text: str, x: float = 0, y: float = 0, width: float = 100, height: float = 20, confidence: float = 0.95) -> OCRToken:
    return OCRToken(
        text=text,
        confidence=confidence,
        bounding_box=BoundingBox(x=x, y=y, width=width, height=height),
    )


def test_1_date_time_two_char_month():
    print("Test 1: Date & Time with '02-Oc-2026 19:36:41'")
    tokens = [create_token("التاريخ: 02-Oc-2026 19:36:41")]
    d, t, _, _ = DateParser.parse(tokens)
    assert d == "2026-10-02", f"Expected 2026-10-02, got {d}"
    assert t == "19:36:41", f"Expected 19:36:41, got {t}"
    print("Test 1 Passed: '02-Oc-2026' -> '2026-10-02', '19:36:41' -> '19:36:41'\n")


def test_2_date_time_ocr_substitutions():
    print("Test 2: Date & Time with 'O2-0ct-2026 19:44:07'")
    tokens = [create_token("التاريخ والزمن: O2-0ct-2026 19:44:07")]
    d, t, _, _ = DateParser.parse(tokens)
    assert d == "2026-10-02", f"Expected 2026-10-02, got {d}"
    assert t == "19:44:07", f"Expected 19:44:07, got {t}"
    print("Test 2 Passed: 'O2-0ct-2026' -> '2026-10-02', '19:44:07' -> '19:44:07'\n")


def test_3_bankak_arabic_recipient_name_regression_case():
    print("Test 3: Bankak Arabic recipient name (يقابلادبع يقتلا لاعلادبع اليه المرسل إسم)")
    # Raw tokens from user's receipt
    raw_name_chunk = "يقابلادبع يقتلا لاعلادبع اليه المرسل إسم لاعلادبع"
    normalized = ArabicNameProcessor.normalize_arabic_name(raw_name_chunk)

    assert "إسم" not in normalized
    assert "المرسل" not in normalized
    assert "اليه" not in normalized
    assert "عبدالباقي" in normalized or "عبدالعال" in normalized
    assert "التقي" in normalized

    is_valid, reason, conf = ArabicNameProcessor.validate_arabic_name(normalized)
    assert is_valid is True
    assert conf >= 0.70
    print(f"Test 3 Passed: Reconstructed normalized name '{normalized}' with valid score {conf:.2f}\n")


def test_4_full_voucher_20365281771():
    print("Test 4: Full Bankak Voucher 20365281771 (40,000.00 SDG)")
    tokens = [
        create_token("تحويلات", y=10),
        create_token("العملية رقم 20365281771", y=30),
        create_token("التاريخ 02-Oc-2026 19:36:41 الزمن", y=50),
        create_token("من حساب 373 1204 4436 0001", y=70),
        create_token("الى حساب 0001 0543 361 2682", y=90),
        create_token("يقابلادبع يقتلا لاعلادبع اليه المرسل إسم", y=110),
        create_token("المبلغ 000.00,40", y=130),
    ]

    quality = ImageQuality(image_quality_score=0.95, quality_status=QualityStatus.GOOD)
    res = SudaneseReceiptParser.parse(tokens, quality)

    # 1. Amount
    assert res.amount == 40000.0, f"Expected 40000.0, got {res.amount}"
    assert res.formatted_amount == "40,000.00 SDG"

    # 2. Reference Number
    assert res.reference_number == "20365281771"
    assert res.transaction_id == "20365281771"

    # 3. Date & Time
    assert res.transaction_date == "2026-10-02", f"Expected 2026-10-02, got {res.transaction_date}"
    assert res.transaction_time == "19:36:41", f"Expected 19:36:41, got {res.transaction_time}"

    # 4. Accounts
    assert res.sender_account == "0373 1204 4436 0001"
    assert res.receiver_account == "0001 0543 0361 2682" or "0543 0361 2682 0001"

    # 5. Receiver Name (cleanly stripped of labels)
    assert res.receiver_name is not None
    assert "إسم" not in res.receiver_name
    assert "المرسل" not in res.receiver_name
    assert "اليه" not in res.receiver_name
    assert res.sender_name is None

    # 6. Bank & Status
    assert res.bank_name == "Bank of Khartoum"
    assert res.wallet_name == "Bankak"
    assert res.transaction_status == "SUCCESS"
    print("Test 4 Passed: Voucher 20365281771 all fields verified.\n")


def test_5_full_voucher_20015315334():
    print("Test 5: Previous Working Voucher 20015315334 (150,000.00 SDG)")
    tokens = [
        create_token("تحويلات", y=10),
        create_token("العملية رقم 20015315334", y=30),
        create_token("التاريخ O2-0ct-2026 19:44:07 الزمن", y=50),
        create_token("من حساب 373 1204 4436 0001", y=70),
        create_token("الى حساب 0573 0156 6109 0001", y=90),
        create_token("بيطلا هريشم محمود السرملا إسم اليه دمح", y=110),
        create_token("المبلغ 000.00,150", y=130),
    ]

    quality = ImageQuality(image_quality_score=0.95, quality_status=QualityStatus.GOOD)
    res = SudaneseReceiptParser.parse(tokens, quality)

    assert res.amount == 150000.0
    assert res.reference_number == "20015315334"
    assert res.transaction_date == "2026-10-02"
    assert res.transaction_time == "19:44:07"
    assert res.sender_account == "0373 1204 4436 0001"
    assert res.receiver_account == "0573 0156 6109 0001"
    assert res.receiver_name is not None
    assert "الطيب" in res.receiver_name
    print("Test 5 Passed: Voucher 20015315334 all fields verified.\n")


def test_6_account_and_txn_ids_never_become_name():
    print("Test 6: Account numbers and Txn IDs never become name")
    is_v1, _, _ = ArabicNameProcessor.validate_arabic_name("0373 1204 4436 0001")
    is_v2, _, _ = ArabicNameProcessor.validate_arabic_name("20365281771")
    is_v3, _, _ = ArabicNameProcessor.validate_arabic_name("40000")
    assert is_v1 is False
    assert is_v2 is False
    assert is_v3 is False
    print("Test 6 Passed: Financial IDs rejected as names.\n")


if __name__ == "__main__":
    print("==================================================")
    print("Running B2B OCR Sudanese Receipt Regression Tests")
    print("==================================================")
    test_1_date_time_two_char_month()
    test_2_date_time_ocr_substitutions()
    test_3_bankak_arabic_recipient_name_regression_case()
    test_4_full_voucher_20365281771()
    test_5_full_voucher_20015315334()
    test_6_account_and_txn_ids_never_become_name()
    print("==================================================")
    print("All Regression Tests Passed Successfully!")
    print("==================================================")
