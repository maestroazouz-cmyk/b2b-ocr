import re
from typing import List, Optional
from app.models.schemas import OCRToken


class RTLNormalizer:
    """
    Handles Arabic text normalization and detects reversed Arabic words from OCR output
    without altering numbers, dates, times, or Latin tokens.
    """

    # Dictionary of common banking labels and names in their reversed forms
    REVERSED_ARABIC_MAP = {
        # Operation / Transaction
        "ةيلمعلا": "العملية",
        "مقر": "رقم",
        "ةيلمعلا مقر": "رقم العملية",
        "ةلماعملا": "المعاملة",
        "يعجرلما": "المرجعي",
        "راشعلإا": "الإشعار",
        "راشعلا": "الإشعار",
        # Date & Time
        "خيراتلا": "التاريخ",
        "نمزلا": "الزمن",
        "نمزلاو": "والزمن",
        "تقولا": "الوقت",
        "نمزلاو خيراتلا": "التاريخ والزمن",
        "خيرات": "تاريخ",
        # Accounts
        "باسح": "حساب",
        "نم باسح": "من حساب",
        "ىلا باسح": "الى حساب",
        "يلا باسح": "الى حساب",
        "حساب لا": "الى حساب",
        "باسح لا": "الى حساب",
        "لا باسح": "الى حساب",
        "حسابلا": "الى حساب",
        "نم": "من",
        "ىلا": "الى",
        "يلا": "الى",
        # Parties & Labels
        "ماسا": "اسم",
        "مسأ": "اسم",
        "مسإ": "إسم",
        "لسرمل": "المرسل",
        "لسرملا": "المرسل",
        "المرسل": "المرسل",
        "هيلا": "اليه",
        "هيلي": "اليه",
        "اليه": "اليه",
        "هيلا لسرمل": "المرسل اليه",
        "هيلا لسرمل ماسا": "اسم المرسل اليه",
        "ديفتسلما": "المستفيد",
        "ديفَتسلما": "المستفيد",
        "ليموعلا": "العميل",
        # Common Name Components (Reversed -> Normal)
        "وذ": "ذو",
        "نونلا": "النون",
        "مشاه": "هاشم",
        "يلع": "علي",
        "دمحا": "احمد",
        "دمحم": "محمد",
        "دومحم": "محمود",
        "للها دبع": "عبد الله",
        "دبع": "عبد",
        "نامثع": "عثمان",
        "نسح": "حسن",
        "نيسح": "حسين",
        # Mobile & Narration
        "ليابوملا": "الموبايل",
        "ليابوم": "الموبايل",
        "ليابوم مقر": "رقم الموبايل",
        "فتاهلا": "الهاتف",
        "قيلعتلا": "التعليق",
        "نايبلا": "البيان",
        "ضرغلا": "الغرض",
        "تاظحلام": "ملاحظات",
        # Amount & Currency
        "غلبملا": "المبلغ",
        "غلبا": "المبلغ",
        "غلUnit": "المبلغ",
        "غل": "المبلغ",
        "مكل": "المبلغ",
        "ةميقلا": "القيمة",
        "هينج": "جنيه",
        "يناودس": "سوداني",
        "يناودس هينج": "جنيه سوداني",
        # Banks
        "موطرخلا": "الخرطوم",
        "كنب": "بنك",
        "موطرخلا كنب": "بنك الخرطوم",
        "ككنبا موطموطرخلا": "بنك الخرطوم",
        "ككنب": "بنكك",
        "تاليوحت": "تحويلات",
        "ليصوت": "تحويل",
        "ليصفت": "تفاصيل",
        "حاجنب": "بنجاح",
        "تم": "تم",
    }

    @classmethod
    def is_arabic_word(cls, text: str) -> bool:
        """Returns True if text consists primarily of Arabic characters."""
        if not text:
            return False
        arabic_chars = re.findall(r"[\u0600-\u06FF]", text)
        return len(arabic_chars) >= max(1, len(text.replace(" ", "")) * 0.5)

    @classmethod
    def reverse_arabic_string(cls, text: str) -> str:
        """Reverses Arabic character sequence if it was CTC-reversed."""
        words = text.split()
        reversed_words = []
        for w in words:
            if cls.is_arabic_word(w):
                # Reverse individual Arabic token
                reversed_words.append(w[::-1])
            else:
                # Keep Latin / Numbers untouched
                reversed_words.append(w)
        # Reverse word sequence for RTL
        return " ".join(reversed_words[::-1])

    @classmethod
    def normalize_token_text(cls, text: str) -> str:
        """
        Normalizes token text:
        1. Checks dictionary replacements for known reversed words
        2. Leaves pure numbers, dates, times, and Latin intact
        """
        if not text:
            return ""

        clean = text.strip()

        # Check exact dictionary match
        if clean in cls.REVERSED_ARABIC_MAP:
            return cls.REVERSED_ARABIC_MAP[clean]

        # Multi-word check
        words = clean.split()
        normalized_words = []
        for w in words:
            if w in cls.REVERSED_ARABIC_MAP:
                normalized_words.append(cls.REVERSED_ARABIC_MAP[w])
            else:
                normalized_words.append(w)

        reconstructed = " ".join(normalized_words)

        if reconstructed in cls.REVERSED_ARABIC_MAP:
            return cls.REVERSED_ARABIC_MAP[reconstructed]

        return reconstructed

    @classmethod
    def normalize_tokens(cls, tokens: List[OCRToken]) -> List[OCRToken]:
        """
        Returns a new list of OCRTokens with normalized text while preserving original bounding boxes.
        """
        normalized_list: List[OCRToken] = []
        for tok in tokens:
            norm_text = cls.normalize_token_text(tok.text)
            new_tok = OCRToken(
                text=norm_text,
                confidence=tok.confidence,
                bounding_box=tok.bounding_box,
                line_index=tok.line_index,
            )
            normalized_list.append(new_tok)
        return normalized_list
