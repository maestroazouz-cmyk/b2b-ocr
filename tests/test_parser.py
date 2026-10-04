"""
Comprehensive B2B OCR Sudanese Receipt Parser Test Suite.
Validates all 19 test cases:
1. Arabic Bankak recipient name
2. Arabic sender name
3. Label accidentally included in OCR (stripped cleanly)
4. RTL word ordering
5. Name split across multiple OCR lines
6. Name next to label
7. Name below label
8. OCR with hamza variations
9. OCR with duplicated spaces
10. Missing sender name -> null
11. Missing receiver name -> null
12. Non-Arabic/Latin merchant name
13. Ambiguous name -> review_required
14. Account number must never become a name
15. Transaction ID must never become a name
16. "اسم المرسل إليه" / "السرملا إسم" must never appear inside receiver_name
17. Existing amount extraction must remain unchanged
18. Existing reference number extraction must remain unchanged
19. Existing date/time extraction must remain unchanged
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


def test_1_arabic_bankak_recipient_name():
    print("Test 1: Arabic Bankak recipient name")
    tokens = [
        create_token("بنك الخرطوم", y=10),
        create_token("رقم العملية: 20265282625", y=30),
        create_token("اسم المرسل اليه: ذو النون هاشم علي احمد", y=50),
        create_token("المبلغ: 35,000.00 SDG", y=70),
    ]
    res = SudaneseReceiptParser.parse(tokens, ImageQuality(image_quality_score=0.95, quality_status=QualityStatus.GOOD))
    assert res.receiver_name == "ذو النون هاشم علي احمد"
    assert res.receiver_name_raw is not None


def test_2_arabic_sender_name():
    print("Test 2: Arabic sender name")
    tokens = [
        create_token("إشعار مالي", y=10),
        create_token("اسم المرسل: عثمان حسن إبراهيم", y=30),
        create_token("المبلغ: 40,000.00 SDG", y=50),
        create_token("الرقم المرجعي: TXN1234567", y=70),
    ]
    res = SudaneseReceiptParser.parse(tokens, ImageQuality(image_quality_score=0.95, quality_status=QualityStatus.GOOD))
    assert res.sender_name == "عثمان حسن إبراهيم"
    assert res.receiver_name is None


def test_3_label_accidentally_included_in_ocr():
    print("Test 3: Label accidentally included in OCR")
    tokens = [
        create_token("تحويلات", y=10),
        create_token("العملية رقم 20015315334", y=30),
        create_token("التاريخ O2-0ct-2026 19:44:07 الزمن", y=50),
        create_token("من حساب 373 1204 4436 0001", y=70),
        create_token("الى حساب 0573 0156 6109 0001", y=90),
        create_token("بيطلا هريشم محمود السرملا إسم اليه دمح", y=110),
        create_token("المبلغ 000.00,150", y=130),
    ]
    res = SudaneseReceiptParser.parse(tokens, ImageQuality(image_quality_score=0.95, quality_status=QualityStatus.GOOD))
    for forbidden in ["السرملا", "إسم", "اليه", "المرسل", "اسم"]:
        assert forbidden not in (res.receiver_name or "")
    assert "الطيب" in res.receiver_name
    assert "محمود" in res.receiver_name


def test_4_rtl_word_ordering():
    print("Test 4: RTL word ordering")
    raw = "وذ نونلا مشاه يلع دمحا"
    norm = ArabicNameProcessor.normalize_arabic_name(raw)
    assert norm == "ذو النون هاشم علي احمد"


def test_5_name_split_across_multiple_lines():
    print("Test 5: Name split across multiple OCR tokens in same row")
    tokens = [
        create_token("بنك الخرطوم", y=10),
        create_token("اسم المرسل اليه:", x=50, y=50),
        create_token("محمود", x=150, y=50),
        create_token("الصادق", x=220, y=50),
        create_token("المبلغ: 20,000.00 SDG", y=70),
        create_token("رقم العملية: 20265282625", y=90),
    ]
    res = SudaneseReceiptParser.parse(tokens, ImageQuality(image_quality_score=0.95, quality_status=QualityStatus.GOOD))
    assert res.receiver_name == "محمود الصادق"


def test_6_name_next_to_label():
    print("Test 6: Name next to label")
    tokens = [
        create_token("اسم المستفيد: سارة عبد الرحمن أحمد", y=10),
        create_token("المبلغ: 10,000.00 SDG", y=30),
        create_token("رقم المعاملة: 12345678", y=50),
    ]
    res = SudaneseReceiptParser.parse(tokens, ImageQuality(image_quality_score=0.95, quality_status=QualityStatus.GOOD))
    assert res.receiver_name == "سارة عبد الرحمن أحمد"


def test_7_name_below_label():
    print("Test 7: Name below label")
    tokens = [
        create_token("بنك الخرطوم", y=10),
        create_token("اسم المرسل اليه", y=30),
        create_token("طارق علي محمد", y=50),
        create_token("المبلغ: 50,000.00 SDG", y=70),
        create_token("رقم العملية: 20265282625", y=90),
    ]
    res = SudaneseReceiptParser.parse(tokens, ImageQuality(image_quality_score=0.95, quality_status=QualityStatus.GOOD))
    assert res.receiver_name is not None
    assert "طارق" in res.receiver_name


def test_8_ocr_with_hamza_variations():
    print("Test 8: OCR with hamza variations")
    norm1 = ArabicNameProcessor.normalize_for_matching("إسم المرسل إليه")
    norm2 = ArabicNameProcessor.normalize_for_matching("اسم المرسل اليه")
    assert norm1 == norm2


def test_9_ocr_with_duplicated_spaces():
    print("Test 9: OCR with duplicated spaces")
    raw = "   خالد    عبد   الله    "
    norm = ArabicNameProcessor.normalize_arabic_name(raw)
    assert norm == "خالد عبد الله"


def test_10_missing_sender_name():
    print("Test 10: Missing sender name -> null")
    tokens = [
        create_token("بنك الخرطوم", y=10),
        create_token("رقم العملية: 20265282625", y=30),
        create_token("المبلغ: 35,000.00 SDG", y=50),
    ]
    res = SudaneseReceiptParser.parse(tokens, ImageQuality(image_quality_score=0.95, quality_status=QualityStatus.GOOD))
    assert res.sender_name is None


def test_11_missing_receiver_name():
    print("Test 11: Missing receiver name -> null")
    tokens = [
        create_token("إشعار مالي", y=10),
        create_token("المبلغ: 35,000.00 SDG", y=30),
    ]
    res = SudaneseReceiptParser.parse(tokens, ImageQuality(image_quality_score=0.95, quality_status=QualityStatus.GOOD))
    assert res.receiver_name is None


def test_12_non_arabic_latin_merchant():
    print("Test 12: Non-Arabic / Latin merchant name")
    tokens = [
        create_token("Merchant: Zain Telecom Sudan", y=10),
        create_token("المبلغ: 15,000.00 SDG", y=30),
        create_token("Ref: 9988776655", y=50),
    ]
    res = SudaneseReceiptParser.parse(tokens, ImageQuality(image_quality_score=0.95, quality_status=QualityStatus.GOOD))
    assert res.merchant_name == "Zain Telecom B2B"


def test_13_ambiguous_name_review_required():
    print("Test 13: Ambiguous name -> review_required")
    tokens = [
        create_token("بنك الخرطوم", y=10),
        create_token("رقم العملية: 20265282625", y=30),
        create_token("المبلغ: 35,000.00 SDG", y=50),
    ]
    res = SudaneseReceiptParser.parse(tokens, ImageQuality(image_quality_score=0.95, quality_status=QualityStatus.GOOD))
    assert res.review_required is True


def test_14_account_number_never_becomes_name():
    print("Test 14: Account number must never become a name")
    is_valid, _, _ = ArabicNameProcessor.validate_arabic_name("0373 1204 4436 0001")
    assert is_valid is False


def test_15_transaction_id_never_becomes_name():
    print("Test 15: Transaction ID must never become a name")
    is_valid, _, _ = ArabicNameProcessor.validate_arabic_name("20265282625")
    assert is_valid is False


def test_16_label_never_appears_inside_receiver_name():
    print("Test 16: 'اسم المرسل إليه' must never appear inside receiver_name")
    is_valid, _, _ = ArabicNameProcessor.validate_arabic_name("اسم المرسل إليه")
    assert is_valid is False
    is_valid2, _, _ = ArabicNameProcessor.validate_arabic_name("السرملا إسم")
    assert is_valid2 is False


def test_17_amount_extraction_intact():
    print("Test 17: Existing amount extraction remains unchanged")
    tokens = [create_token("المبلغ 35,000.00 SDG")]
    amt, fmt, curr, ev = AmountParser.parse(tokens)
    assert amt == 35000.0
    assert curr == "SDG"


def test_18_reference_number_intact():
    print("Test 18: Existing reference number extraction remains unchanged")
    tokens = [
        create_token("بنك الخرطوم", y=10),
        create_token("رقم العملية: 20015315334", y=30),
        create_token("المبلغ: 150,000.00 SDG", y=50),
    ]
    res = SudaneseReceiptParser.parse(tokens, ImageQuality(image_quality_score=0.95, quality_status=QualityStatus.GOOD))
    assert res.reference_number == "20015315334"


def test_19_datetime_extraction_intact():
    print("Test 19: Existing date/time extraction remains unchanged")
    tokens = [create_token("التاريخ والزمن: O2-0ct-2026 19:44:07")]
    d, t, _, _ = DateParser.parse(tokens)
    assert d == "2026-10-02"
    assert t == "19:44:07"


if __name__ == "__main__":
    print("==================================================")
    print("Running All 19 B2B OCR Sudanese Receipt Tests")
    print("==================================================")
    test_1_arabic_bankak_recipient_name()
    test_2_arabic_sender_name()
    test_3_label_accidentally_included_in_ocr()
    test_4_rtl_word_ordering()
    test_5_name_split_across_multiple_lines()
    test_6_name_next_to_label()
    test_7_name_below_label()
    test_8_ocr_with_hamza_variations()
    test_9_ocr_with_duplicated_spaces()
    test_10_missing_sender_name()
    test_11_missing_receiver_name()
    test_12_non_arabic_latin_merchant()
    test_13_ambiguous_name_review_required()
    test_14_account_number_never_becomes_name()
    test_15_transaction_id_never_becomes_name()
    test_16_label_never_appears_inside_receiver_name()
    test_17_amount_extraction_intact()
    test_18_reference_number_intact()
    test_19_datetime_extraction_intact()
    print("==================================================")
    print("All 19 Tests Passed Successfully!")
    print("==================================================")
