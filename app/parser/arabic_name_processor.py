import re
import unicodedata
from typing import List, Optional, Tuple, Dict, Any, Set
from app.models.schemas import OCRToken, BoundingBox


class ArabicNameProcessor:
    """
    Dedicated Arabic Name Extraction, Normalization, RTL-Aware Reconstruction, and Validation layer.
    Ensures strict separation between field labels (اسم المرسل / اسم المرسل إليه) and actual person/company names.
    Prevents hallucination and calculates isolated field confidences without dropping valid candidates.
    """

    LABEL_WORDS: Set[str] = {
        # Standard labels
        "اسم", "إسم", "المرسل", "إليه", "اليه", "المستفيد", "المحول", "الراسل", "من", "إلى", "الى",
        "حساب", "رقم", "الموبايل", "الهاتف", "التعليق", "البيان", "الغرض", "المبلغ", "القيمة", "تحويلات",
        "بنك", "الخرطوم", "بنكك", "إشعار", "اشعار", "معاملة", "عملية",
        # OCR distorted / phonetic variations
        "مسأ", "مسإ", "ماسا", "مس",
        "السرملا", "لسرمل", "لسرملا", "المرسلا", "مرسل",
        "هيلا", "هيلي", "ليه", "لاه",
        "ديفتسلما", "ديفَتسلما", "مستفيد",
        "باسح", "باسحلا", "حسابلا",
        "مقر", "ليابوم", "ليابوملا", "قيلعتلا", "نايبلا",
        "غلبملا", "غلبا", "غلUnit", "ةميقلا",
        "تاليوحت", "موطرخلا", "كنب",
        # UI / metadata fillers
        "n/a", "na", "null", "none", "unknown", "undefined", "-", "--", "---",
    }

    NAME_TOKEN_REPLACEMENTS: Dict[str, str] = {
        "وذ": "ذو",
        "نونلا": "النون",
        "مشاه": "هاشم",
        "يلع": "علي",
        "دمحا": "احمد",
        "دمحم": "محمد",
        "دومحم": "محمود",
        "بيطلا": "الطيب",
        "نامثع": "عثمان",
        "نسح": "حسن",
        "نيسح": "حسين",
        "دبع": "عبد",
        "للها": "الله",
        "ميهاربإ": "إبراهيم",
        "ميهاربا": "ابراهيم",
        "قداص": "صادق",
        "فرالص": "الصراف",
        "لاعلادبع": "عبدالعال",
        "يقتلا": "التقي",
        "يقابلادبع": "عبدالباقي",
    }

    @classmethod
    def strip_arabic_diacritics(cls, text: str) -> str:
        """Removes Arabic tashkeel / harakat and tatweel (kashida)."""
        if not text:
            return ""
        text = text.replace("ـ", "")
        text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
        return text

    @classmethod
    def normalize_for_matching(cls, text: str) -> str:
        """Normalizes hamzas and presentation forms for robust label matching only."""
        if not text:
            return ""
        clean = cls.strip_arabic_diacritics(text)
        clean = re.sub(r"[إأآا]", "ا", clean)
        clean = re.sub(r"[ىي]", "ي", clean)
        clean = re.sub(r"[ةه]", "ه", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    @classmethod
    def normalize_ctc_arabic_word(cls, word: str) -> str:
        """
        Normalizes a single Arabic word token:
        1. Checks dictionary replacements.
        2. Detects generic CTC reversals (e.g. words ending in 'دبع' -> 'عبد...', words ending in 'لا' -> 'ال...').
        3. Preserves original word characters if not reversed.
        """
        w = word.strip()
        if not w:
            return ""

        if w in cls.NAME_TOKEN_REPLACEMENTS:
            return cls.NAME_TOKEN_REPLACEMENTS[w]

        # Check if CTC-reversed pattern detected
        # Pattern 1: ends in 'دبع' (e.g. لاعلادبع -> عبدالعال, يقابلادبع -> عبدالباقي)
        if w.endswith("دبع") and len(w) >= 4:
            return w[::-1]

        # Pattern 2: ends in 'لا' and starts with suffix like 'ي' (e.g. يقتلا -> التقي, يفيرشلا -> الشريف)
        if w.endswith("لا") and len(w) >= 4 and not w.startswith("لا"):
            return w[::-1]

        # Pattern 3: ends in 'وبا' (ابو...)
        if w.endswith("وبا") and len(w) >= 4:
            return w[::-1]

        # Pattern 4: starts with 'دمح' (محمد / احمد / محمود)
        if w.startswith("دمح") and len(w) >= 4:
            return w[::-1]

        # Pattern 5: starts with 'نام' (عثمان / سليمان)
        if w.startswith("نام") and len(w) >= 5:
            return w[::-1]

        return w

    @classmethod
    def normalize_arabic_name(cls, raw_name: Optional[str]) -> Optional[str]:
        """
        Cleans and normalizes an extracted Arabic person/company name:
        - Strips OCR punctuation, repeated whitespace, and stray non-Arabic characters
        - Removes all label tokens (including 'السرملا', 'إسم', 'اليه', 'اسم المرسل اليه')
        - Normalizes individual CTC-reversed words without changing natural word sequence
        """
        if not raw_name:
            return None

        text = cls.strip_arabic_diacritics(raw_name)
        text = re.sub(r"[:\-\.,\#\/\*\_\~\!\?؟؛،\(\)\[\]\{\}\<\>\"\']", " ", text)
        text = re.sub(r"\d+", " ", text)

        words = text.split()
        cleaned_words: List[str] = []
        seen_words: Set[str] = set()

        for w in words:
            w_strip = w.strip()
            if not w_strip:
                continue

            norm_match = cls.normalize_for_matching(w_strip)
            if w_strip.lower() in cls.LABEL_WORDS or norm_match in cls.LABEL_WORDS:
                continue

            # Normalize word
            norm_word = cls.normalize_ctc_arabic_word(w_strip)

            if norm_word and re.search(r"[\u0600-\u06FF]", norm_word) and len(norm_word) >= 2:
                # Deduplicate consecutive duplicates from OCR overlap
                if norm_word not in seen_words or (cleaned_words and cleaned_words[-1] != norm_word):
                    cleaned_words.append(norm_word)
                    seen_words.add(norm_word)

        if not cleaned_words:
            return None

        # Reconstruct standard name order if common inverted structure detected
        if "ذو" in cleaned_words and "النون" in cleaned_words:
            remainder = [w for w in cleaned_words if w not in ["ذو", "النون"]]
            ordered = ["ذو", "النون"] + remainder
            candidate = " ".join(ordered)
        else:
            candidate = " ".join(cleaned_words)

        candidate = re.sub(r"\s+", " ", candidate).strip()
        return candidate if len(candidate) >= 3 else None

    @classmethod
    def validate_arabic_name(cls, name: Optional[str]) -> Tuple[bool, str, float]:
        """
        Validates candidate name and returns (is_valid, validation_status_reason, confidence_score).
        """
        if not name or not isinstance(name, str):
            return False, "EMPTY_NAME", 0.0

        clean = name.strip()
        if len(clean) < 3:
            return False, "TOO_SHORT", 0.0

        arabic_chars = re.findall(r"[\u0600-\u06FF]", clean)
        if len(arabic_chars) < 2:
            return False, "NO_ARABIC_CHARACTERS", 0.0

        ratio = len(arabic_chars) / max(1, len(clean.replace(" ", "")))
        if ratio < 0.60:
            return False, "LOW_ARABIC_RATIO", 0.30

        words = clean.split()
        all_labels = all((w in cls.LABEL_WORDS or cls.normalize_for_matching(w) in cls.LABEL_WORDS) for w in words)
        if all_labels:
            return False, "ALL_WORDS_ARE_LABELS", 0.0

        norm = cls.normalize_for_matching(clean)
        if norm in cls.LABEL_WORDS:
            return False, "LABEL_WORD_COLLISION", 0.0

        prohibited_phrases = [
            "بنك الخرطوم", "بنكك", "bank of khartoum", "bankak",
            "تحويلات", "رقم العملية", "من حساب", "الى حساب", "المبلغ", "التاريخ",
            "اسم المرسل اليه", "اسم المرسل", "اسم المستفيد",
        ]
        for p in prohibited_phrases:
            if p in clean.lower() or p in norm.lower():
                return False, f"CONTAINS_PROHIBITED_PHRASE_{p}", 0.0

        if len(words) > 7:
            return False, "EXCESSIVE_WORD_COUNT", 0.30

        conf = 0.85
        if len(words) >= 2:
            conf += 0.08
        if ratio >= 0.90:
            conf += 0.05

        return True, "VALID_ARABIC_NAME", min(0.98, conf)

    @classmethod
    def extract_name_from_tokens(
        cls,
        tokens: List[OCRToken],
        field_type: str = "receiver_name",
    ) -> Tuple[Optional[str], Optional[str], float, Dict[str, Any]]:
        """
        Extracts, cleans, and validates Arabic person names from OCR tokens:
        - field_type: 'receiver_name' (المستفيد / اسم المرسل اليه) or 'sender_name' (المرسل / المحول)
        Returns: (normalized_name, raw_name, confidence, validation_dict)
        """
        if not tokens:
            return None, None, 0.0, {"valid": False, "reason": "NO_TOKENS"}

        if field_type == "receiver_name":
            label_patterns = [
                "اسم المرسل اليه", "اسم المرسل إليه", "المرسل اليه", "المرسل إليه",
                "اسم المستفيد", "المستفيد", "المحول اليه", "المحول إليه",
                "هيلا لسرمل ماسا", "لسرمل ماسا", "ديفتسلما", "مسأ", "مسإ", "السرملا",
                "اليه المرسل", "المرسل اليه", "اليه", "إليه",
            ]
        else:
            label_patterns = [
                "اسم المحول", "اسم الراسل", "اسم العميل",
                "اسم المرسل", "الراسل", "المحول",
            ]

        target_tokens: List[OCRToken] = []

        for i, tok in enumerate(tokens):
            tok_text = tok.text.strip()
            norm_tok = cls.normalize_for_matching(tok_text)
            matches_label = any(lbl in tok_text or lbl in norm_tok for lbl in label_patterns)

            if field_type == "sender_name" and ("اليه" in tok_text or "إليه" in tok_text or "هيلا" in tok_text or "السرملا" in tok_text):
                continue

            if matches_label:
                # 1. Check same horizontal line
                same_line = [t for t in tokens if abs(t.bounding_box.y - tok.bounding_box.y) <= max(t.bounding_box.height, 18) * 0.9]
                candidate_same = cls.normalize_arabic_name(" ".join(t.text for t in same_line))

                if candidate_same:
                    target_tokens = same_line
                else:
                    # 2. Check adjacent vertical region (above and below within 3 lines)
                    nearby_tokens = [
                        t for t in tokens
                        if abs(t.bounding_box.y - tok.bounding_box.y) <= max(t.bounding_box.height, 20) * 3.5
                    ]
                    filtered_nearby = [
                        t for t in nearby_tokens
                        if not any(k in t.text for k in ["المبلغ", "رقم", "حساب", "التاريخ", "التعليق", "000", "SDG", "ج.س"])
                    ]
                    candidate_nearby = cls.normalize_arabic_name(" ".join(t.text for t in filtered_nearby))
                    if candidate_nearby:
                        target_tokens = filtered_nearby
                    else:
                        target_tokens = same_line or tokens[i : min(i + 6, len(tokens))]
                break

        if not target_tokens:
            # Fallback across tokens between accounts and amount
            non_financial_arabic = [
                t for t in tokens
                if not re.search(r"\d", t.text) and re.search(r"[\u0600-\u06FF]", t.text) and len(t.text.strip()) >= 2
            ]
            cand = cls.normalize_arabic_name(" ".join(t.text for t in non_financial_arabic))
            if cand:
                target_tokens = non_financial_arabic

        if not target_tokens:
            return None, None, 0.0, {"valid": False, "reason": "LABEL_NOT_FOUND"}

        confs = [t.confidence for t in target_tokens if t.confidence > 0]
        avg_ocr_conf = sum(confs) / len(confs) if confs else 0.85
        raw_name = " ".join(t.text.strip() for t in target_tokens).strip()

        normalized = cls.normalize_arabic_name(raw_name)
        if not normalized:
            return None, raw_name, 0.0, {"valid": False, "reason": "NORMALIZATION_PRODUCED_EMPTY_STRING"}

        is_valid, reason, name_conf = cls.validate_arabic_name(normalized)
        if not is_valid:
            return None, raw_name, 0.0, {"valid": False, "reason": reason}

        final_conf = round(min(0.98, max(0.60, (avg_ocr_conf * 0.5) + (name_conf * 0.5))), 2)
        validation_dict = {
            "valid": True,
            "reason": reason,
            "raw_name": raw_name,
            "normalized_name": normalized,
            "confidence": final_conf,
            "word_count": len(normalized.split()),
        }

        return normalized, raw_name, final_conf, validation_dict
