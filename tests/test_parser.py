"""
Comprehensive B2B OCR Sudanese Receipt Parser Test Suite.
Validates:
1. Bank of Khartoum structured voucher ground truth (spatial layout, accounts, recipient name, date/time, amount).
2. OCR error normalization: "O3-0ct-2026" -> "2026-10-03".
3. Account leading/trailing zero padding: "373 1204 4436 0001" -> "0373 1204 4436 0001".
4. Receiver account reconstruction from fragmented OCR with label noise: "0913", "حساب لا", "0833", "1734", "001".
5. Arabic recipient name reconstruction from reversed OCR tokens: "يلع", "مشاه", "مسأ", "نونلا", "اليه", "المرسل", "دمحا", "وذ" -> "ذو النون هاشم علي احمد".
6. Overall confidence degradation when fields are missing (cannot remain 0.98!).
7. Protection against transaction numbers / account numbers being misinterpreted as amounts.
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
from app.parser.date_parser import DateParser


def create_token(text: str, x: float = 0, y: float = 0, width: float = 100, height: float = 20, confidence: float = 0.95) -> OCRToken:
    return OCRToken(
        text=text,
        confidence=confidence,
        bounding_box=BoundingBox(x=x, y=y, width=width, height=height),
    )


def test_bank_of_khartoum_real_receipt_ground_truth():
    print("--- Test 1: Bank of Khartoum Real Receipt Ground Truth with OCR Anomalies ---")
    tokens = [
        create_token("بنك الخرطوم", x=200, y=30),
        create_token("تحويلات", x=220, y=55),
        # Row 1: Transaction Number
        create_token("رقم العملية", x=50, y=100),
        create_token("20265282625", x=250, y=100),
        # Row 2: Date & Time (OCR letters: O3-0ct-2026)
        create_token("التاريخ والزمن", x=50, y=140),
        create_token("O3-0ct-2026", x=250, y=140),
        create_token("10:52:07", x=350, y=140),
        # Row 3: From Account (missing leading zero: 373)
        create_token("من حساب", x=50, y=180),
        create_token("373 1204 4436 0001", x=250, y=180),
        # Row 4: To Account (OCR noise: حساب لا, missing trailing zero: 001)
        create_token("حساب لا", x=50, y=220),
        create_token("0913 0833 1734 001", x=250, y=220),
        # Row 5: Recipient Name (OCR reversed tokens: وذ, نونلا, مشاه, يلع, دمحا)
        create_token("اسم المرسل اليه", x=50, y=260),
        create_token("وذ نونلا مشاه يلع دمحا", x=250, y=260),
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
    assert result.sender_account == "0373 1204 4436 0001", f"Expected '0373 1204 4436 0001', got {result.sender_account}"
    assert result.receiver_account == "0913 0833 1734 0001", f"Expected '0913 0833 1734 0001', got {result.receiver_account}"

    # 5. Names
    assert result.receiver_name == "ذو النون هاشم علي احمد", f"Expected 'ذو النون هاشم علي احمد', got {result.receiver_name}"
    assert result.sender_name is None, f"Sender name must be null on this receipt format, got {result.sender_name}"

    # 6. Bank & Status
    assert result.bank_name == "Bank of Khartoum", f"Expected Bank of Khartoum, got {result.bank_name}"
    assert result.transaction_type == "BANK_TRANSFER"
    assert result.result_status == ExtractionStatus.SUCCESS
    assert result.review_required is False
    assert result.overall_confidence >= 0.90, f"Expected overall confidence >= 0.90, got {result.overall_confidence}"

    print("Test 1 Passed: Real Bank of Khartoum receipt parsed with all 8 target fields verified.\n")


def test_date_ocr_normalization():
    print("--- Test 2: Date OCR Normalization ('O3-0ct-2026' -> '2026-10-03') ---")
    tokens = [create_token("التاريخ والزمن: O3-0ct-2026 10:52:07")]
    date_val, time_val, ev, meta = DateParser.parse(tokens)
    assert date_val == "2026-10-03", f"Expected '2026-10-03', got {date_val}"
    assert time_val == "10:52:07", f"Expected '10:52:07', got {time_val}"
    print("Test 2 Passed: 'O3-0ct-2026' normalized to '2026-10-03'.\n")


def test_sender_and_receiver_account_reconstruction():
    print("--- Test 3: Account Number Zero-Padding & Reconstruction ---")
    tokens = [
        create_token("بنك الخرطوم", y=10),
        create_token("رقم العملية: 20265282625", y=30),
        create_token("من حساب: 373 1204 4436 0001", y=50),
        create_token("حساب لا: 0913 0833 1734 001", y=70),
        create_token("المبلغ: 35,000.00 SDG", y=90),
    ]
    quality = ImageQuality(image_quality_score=0.95, quality_status=QualityStatus.GOOD)
    result = SudaneseReceiptParser.parse(tokens, quality)

    assert result.sender_account == "0373 1204 4436 0001", f"Expected padded sender acc, got {result.sender_account}"
    assert result.receiver_account == "0913 0833 1734 0001", f"Expected padded receiver acc, got {result.receiver_account}"
    print("Test 3 Passed: Sender and receiver account numbers correctly reconstructed.\n")


def test_arabic_receiver_name_reconstruction():
    print("--- Test 4: Arabic Receiver Name Reconstruction from Reversed Tokens ---")
    # Tokens as detected on Render: ylec, mshah, msa', nwnla, alyh, almrsl, dmha, wdh
    tokens = [
        create_token("بنك الخرطوم", y=10),
        create_token("رقم العملية: 20265282625", y=30),
        create_token("مسأ", x=50, y=50),
        create_token("المرسل", x=80, y=50),
        create_token("اليه", x=120, y=50),
        create_token("وذ", x=160, y=50),
        create_token("نونلا", x=190, y=50),
        create_token("مشاه", x=230, y=50),
        create_token("يلع", x=270, y=50),
        create_token("دمحا", x=310, y=50),
        create_token("المبلغ: 35,000.00 SDG", y=70),
    ]
    quality = ImageQuality(image_quality_score=0.95, quality_status=QualityStatus.GOOD)
    result = SudaneseReceiptParser.parse(tokens, quality)

    assert result.receiver_name == "ذو النون هاشم علي احمد", f"Expected 'ذو النون هاشم علي احمد', got {result.receiver_name}"
    assert result.sender_name is None, f"Sender name must be null, got {result.sender_name}"
    print("Test 4 Passed: Recipient name reconstructed accurately from reversed OCR fragments.\n")


def test_confidence_penalty_on_missing_fields():
    print("--- Test 5: Overall Confidence Degradation on Missing Fields ---")
    # Only amount and reference present; date, receiver account, receiver name are missing
    tokens = [
        create_token("بنك الخرطوم", y=10),
        create_token("رقم العملية: 20265282625", y=30),
        create_token("من حساب: 0373 1204 4436 0001", y=50),
        create_token("المبلغ: 35,000.00 SDG", y=70),
    ]
    quality = ImageQuality(image_quality_score=0.95, quality_status=QualityStatus.GOOD)
    result = SudaneseReceiptParser.parse(tokens, quality)

    # Cannot be 0.98! Must drop significantly because date, receiver account, receiver name are missing.
    assert result.overall_confidence < 0.75, f"Expected overall confidence < 0.75, got {result.overall_confidence}"
    assert result.review_required is True, "review_required must be True when major fields are missing"
    print(f"Test 5 Passed: Confidence penalized appropriately ({result.overall_confidence:.2f}) with review_required=True.\n")


def test_amount_safety_against_transaction_numbers():
    print("--- Test 6: Amount Safety - Long Numbers & Transaction IDs Never Become Amount ---")
    tokens = [
        create_token("إشعار مالي", y=10),
        create_token("رقم العملية: 20265282625", y=30),
        create_token("الحساب: 0373 1204 4436 0001", y=50),
    ]
    quality = ImageQuality(image_quality_score=0.85, quality_status=QualityStatus.ACCEPTABLE)
    result = SudaneseReceiptParser.parse(tokens, quality)

    assert result.amount is None, f"Amount must be None when label is missing, got {result.amount}"
    assert result.review_required is True, "Review must be required when amount is unconfirmed"
    print("Test 6 Passed: Transaction number 20265282625 correctly rejected as amount.\n")


if __name__ == "__main__":
    print("==================================================")
    print("Running B2B OCR Sudanese Receipt Parser Test Suite")
    print("==================================================")
    test_bank_of_khartoum_real_receipt_ground_truth()
    test_date_ocr_normalization()
    test_sender_and_receiver_account_reconstruction()
    test_arabic_receiver_name_reconstruction()
    test_confidence_penalty_on_missing_fields()
    test_amount_safety_against_transaction_numbers()
    print("==================================================")
    print("All 6 Focused Unit Tests Passed Successfully!")
    print("==================================================")
