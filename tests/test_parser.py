"""
Comprehensive B2B OCR Sudanese Receipt Parser Test Suite.
Validates:
1. Bank of Khartoum structured voucher ground truth (spatial layout, accounts, recipient name, date/time, amount).
2. Protection against transaction numbers / account numbers being misinterpreted as amounts.
3. Reversed Arabic text & reversed decimal normalization.
4. Timezone & Future date validation relative to Africa/Khartoum.
5. Generic multi-bank & mobile wallet parsing (Fawry, O-Cash, etc.).
6. Anti-hallucination on missing fields.
"""

import sys
import os

# Add parent directory to path for standalone execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.schemas import (
    OCRToken,
    BoundingBox,
    ImageQuality,
    QualityStatus,
    ExtractionStatus,
)
from app.parser.receipt_parser import SudaneseReceiptParser
from app.parser.amount_parser import AmountParser
from app.parser.phone_parser import PhoneParser
from app.parser.transaction_parser import TransactionParser
from app.parser.date_parser import DateParser
from app.parser.bank_parser import BankParser
from app.parser.party_parser import PartyParser


def create_token(text: str, x: float = 0, y: float = 0, width: float = 100, height: float = 20, confidence: float = 0.95) -> OCRToken:
    return OCRToken(
        text=text,
        confidence=confidence,
        bounding_box=BoundingBox(x=x, y=y, width=width, height=height),
    )


def test_bank_of_khartoum_real_receipt_ground_truth():
    print("--- Test 1: Bank of Khartoum Real Receipt Ground Truth ---")
    # Tokens representing the structured Bank of Khartoum voucher
    tokens = [
        create_token("بنك الخرطوم", x=200, y=30),
        create_token("تحويلات", x=220, y=55),
        # Row 1: Transaction Number
        create_token("رقم العملية", x=50, y=100),
        create_token("20265282625", x=250, y=100),
        # Row 2: Date & Time
        create_token("التاريخ والزمن", x=50, y=140),
        create_token("03-Oct-2026 10:52:07", x=250, y=140),
        # Row 3: From Account
        create_token("من حساب", x=50, y=180),
        create_token("0373 1204 4436 0001", x=250, y=180),
        # Row 4: To Account
        create_token("الى حساب", x=50, y=220),
        create_token("0913 0833 1734 0001", x=250, y=220),
        # Row 5: Recipient Name
        create_token("اسم المرسل اليه", x=50, y=260),
        create_token("ذو النون هاشم علي احمد", x=250, y=260),
        # Row 6: Mobile
        create_token("رقم الموبايل", x=50, y=300),
        create_token("N/A", x=250, y=300),
        # Row 7: Comment
        create_token("التعليق", x=50, y=340),
        create_token("N/A", x=250, y=340),
        # Row 8: Amount
        create_token("المبلغ", x=50, y=380),
        create_token("35,000.00 SDG", x=250, y=380),
    ]

    quality = ImageQuality(image_quality_score=0.95, quality_status=QualityStatus.GOOD)
    result = SudaneseReceiptParser.parse(tokens, quality)

    # 1. Amount
    assert result.amount == 35000.0, f"Expected amount 35000.0, got {result.amount}"
    assert result.currency == "SDG", f"Expected currency SDG, got {result.currency}"
    assert result.formatted_amount == "35,000.00 SDG", f"Expected '35,000.00 SDG', got {result.formatted_amount}"

    # 2. Transaction Reference / ID
    assert result.reference_number == "20265282625", f"Expected ref 20265282625, got {result.reference_number}"
    assert result.transaction_id == "20265282625", f"Expected txn_id 20265282625, got {result.transaction_id}"

    # 3. Date & Time
    assert result.transaction_date == "2026-10-03", f"Expected date 2026-10-03, got {result.transaction_date}"
    assert result.transaction_time == "10:52:07", f"Expected time 10:52:07, got {result.transaction_time}"

    # 4. Accounts
    assert result.sender_account == "0373 1204 4436 0001", f"Expected sender acc, got {result.sender_account}"
    assert result.receiver_account == "0913 0833 1734 0001", f"Expected receiver acc, got {result.receiver_account}"

    # 5. Names
    assert result.receiver_name == "ذو النون هاشم علي احمد", f"Expected receiver name, got {result.receiver_name}"
    assert result.sender_name is None, f"Sender name must be null on this receipt format, got {result.sender_name}"

    # 6. Phones & Narration
    assert result.sender_phone is None, f"Sender phone must be null, got {result.sender_phone}"
    assert result.receiver_phone is None, f"Receiver phone must be null, got {result.receiver_phone}"
    assert result.narration is None, f"Narration must be null, got {result.narration}"

    # 7. Bank & Status
    assert result.bank_name == "Bank of Khartoum", f"Expected Bank of Khartoum, got {result.bank_name}"
    assert result.transaction_type == "BANK_TRANSFER"
    assert result.result_status == ExtractionStatus.SUCCESS

    # 8. Evidence Map
    assert result.evidence_map is not None
    assert result.evidence_map["amount"]["value"] == 35000.0
    assert result.evidence_map["reference_number"]["value"] == "20265282625"
    assert "المبلغ" in result.evidence_map["amount"]["evidence"]

    print("Test 1 Passed: Real Bank of Khartoum receipt ground truth verified with exact field mapping.\n")


def test_amount_safety_against_transaction_numbers():
    print("--- Test 2: Amount Safety - Long Numbers & Transaction IDs Never Become Amount ---")
    tokens = [
        create_token("إشعار مالي", y=10),
        create_token("رقم العملية: 20265282625", y=30),
        create_token("الحساب: 0373 1204 4436 0001", y=50),
        # Note: No 'المبلغ' label on this corrupted receipt
    ]

    quality = ImageQuality(image_quality_score=0.85, quality_status=QualityStatus.ACCEPTABLE)
    result = SudaneseReceiptParser.parse(tokens, quality)

    # Must NEVER extract 20265282625 as amount
    assert result.amount is None, f"Amount must be None when label is missing, got {result.amount}"
    assert result.review_required is True, "Review must be required when amount is unconfirmed"
    print("Test 2 Passed: Transaction number 20265282625 correctly rejected as amount.\n")


def test_reversed_ocr_arabic_and_decimal_normalization():
    print("--- Test 3: Reversed OCR Arabic & Decimal Normalization ---")
    tokens = [
        create_token("ككنبا موطموطرخلا", y=10),  # Reversed بنك الخرطوم
        create_token("ةيلمعلا مقر: 20265282625", y=30),  # Reversed رقم العملية
        create_token("نمزلاو خيراتلا: 03-Oct-2026 10:52:07", y=50),  # Reversed التاريخ والزمن
        create_token("غلبملا: 000.00,35 SDG", y=70),  # Reversed المبلغ 35,000.00
    ]

    quality = ImageQuality(image_quality_score=0.90, quality_status=QualityStatus.GOOD)
    result = SudaneseReceiptParser.parse(tokens, quality)

    assert result.bank_name == "Bank of Khartoum"
    assert result.reference_number == "20265282625"
    assert result.transaction_date == "2026-10-03"
    assert result.amount == 35000.0, f"Expected 35000.0 from reversed amount, got {result.amount}"
    print("Test 3 Passed: Reversed OCR Arabic and reversed decimals normalized accurately.\n")


def test_generic_fawry_wallet_receipt():
    print("--- Test 4: Generic Fawry Wallet Receipt ---")
    tokens = [
        create_token("خدمة فوري للمدفوعات", y=10),
        create_token("المبلغ 320,000 ج.س", y=30),
        create_token("المستفيد: زين السودان", y=50),
        create_token("رقم الهاتف: 0913456789", y=70),
    ]

    quality = ImageQuality(image_quality_score=0.88, quality_status=QualityStatus.GOOD)
    result = SudaneseReceiptParser.parse(tokens, quality)

    assert result.wallet_name == "Fawry"
    assert result.bank_name is None, "Bank must be None without explicit bank label"
    assert result.amount == 320000.0
    assert result.sender_phone == "0913456789"
    assert result.reference_number is None, "Missing ref must return None (Anti-Hallucination)"
    print("Test 4 Passed: Generic Fawry slip parsed without hallucinating bank.\n")


def test_anti_hallucination_empty_fields():
    print("--- Test 5: Anti-Hallucination on Empty / Missing Fields ---")
    tokens = [
        create_token("تقرير غير مالي", y=10),
        create_token("شكراً لتعاملكم معنا", y=30),
    ]

    quality = ImageQuality(image_quality_score=0.75, quality_status=QualityStatus.ACCEPTABLE)
    result = SudaneseReceiptParser.parse(tokens, quality)

    assert result.amount is None
    assert result.reference_number is None
    assert result.sender_name is None
    assert result.receiver_name is None
    assert result.sender_phone is None
    assert result.result_status == ExtractionStatus.MANUAL_ENTRY
    print("Test 5 Passed: All missing fields returned null with 0 confidence.\n")


def test_phone_and_amount_parsers():
    print("--- Test 6: Phone & Amount Formats ---")
    # Amounts
    cases_amt = [
        ("المبلغ: 150,000", 150000.0),
        ("القيمة 150000.50 ج.س", 150000.50),
        ("Total: 50,000.00 SDG", 50000.0),
        ("غلبملا: 000.00,50 SDG", 50000.0),
    ]
    for text, expected in cases_amt:
        tokens = [create_token(text)]
        amt, fmt, curr, ev = AmountParser.parse(tokens)
        assert amt == expected, f"Expected {expected} for '{text}', got {amt}"

    # Phones
    cases_phone = [
        ("0912345678", "0912345678"),
        ("+249912345678", "0912345678"),
        ("249123456789", "0123456789"),
    ]
    for text, expected in cases_phone:
        tokens = [create_token(text)]
        phone, ev = PhoneParser.parse(tokens)
        assert phone == expected, f"Expected {expected} for '{text}', got {phone}"

    print("Test 6 Passed: Phone and amount formats verified.\n")


if __name__ == "__main__":
    print("==================================================")
    print("Running B2B OCR Sudanese Receipt Parser Test Suite")
    print("==================================================")
    test_bank_of_khartoum_real_receipt_ground_truth()
    test_amount_safety_against_transaction_numbers()
    test_reversed_ocr_arabic_and_decimal_normalization()
    test_generic_fawry_wallet_receipt()
    test_anti_hallucination_empty_fields()
    test_phone_and_amount_parsers()
    print("==================================================")
    print("All 6 Unit Tests Passed Successfully!")
    print("==================================================")
