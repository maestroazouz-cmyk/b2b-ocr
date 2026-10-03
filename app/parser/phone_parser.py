import re
from typing import List, Optional, Tuple
from app.models.schemas import OCRToken, EvidenceField


class PhoneParser:
    """
    Extracts Sudanese mobile telephone numbers from vouchers:
    - 09XXXXXXXX (Zain Sudan)
    - 01XXXXXXXX (Sudani / MTN)
    - +2499XXXXXXXX / 2499XXXXXXXX
    - +2491XXXXXXXX / 2491XXXXXXXX
    Normalizes into clean standard 10-digit national format (e.g. 0912345678).
    """

    ARABIC_INDIC_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

    PHONE_PATTERNS = [
        r"(?:(?:\+?249|00249)|0)?(9\d{8})(?!\d)",  # Zain: 09XXXXXXXX / +2499XXXXXXXX
        r"(?:(?:\+?249|00249)|0)?(1\d{8})(?!\d)",  # Sudani/MTN: 01XXXXXXXX / +2491XXXXXXXX
    ]

    PHONE_KEYWORDS = ["هاتف", "موبايل", "جوال", "رقم الهاتف", "رقم الموبايل", "Phone", "Mobile", "Tel"]

    @classmethod
    def normalize_digits(cls, text: str) -> str:
        if not text:
            return ""
        return text.translate(cls.ARABIC_INDIC_MAP)

    @classmethod
    def parse(cls, tokens: List[OCRToken]) -> Tuple[Optional[str], EvidenceField]:
        """
        Extracts sender/client phone number from tokens.
        """
        # Pass 1: Proximity to phone keywords
        for i, token in enumerate(tokens):
            norm_text = cls.normalize_digits(token.text)
            has_kw = any(kw in token.text for kw in cls.PHONE_KEYWORDS)

            if has_kw:
                # Check same token
                phone = cls._extract_phone(norm_text)
                if phone:
                    return phone, EvidenceField(
                        value=phone,
                        confidence=min(0.98, token.confidence),
                        source_text=token.text,
                        evidence=f"Labeled phone: '{token.text}'",
                    )

                # Check adjacent tokens
                for j in range(i + 1, min(i + 3, len(tokens))):
                    adj_token = tokens[j]
                    adj_norm = cls.normalize_digits(adj_token.text)
                    phone = cls._extract_phone(adj_norm)
                    if phone:
                        conf = min(0.96, (token.confidence + adj_token.confidence) / 2.0)
                        return phone, EvidenceField(
                            value=phone,
                            confidence=conf,
                            source_text=f"{token.text} -> {adj_token.text}",
                            evidence=f"Phone keyword '{token.text}' followed by '{adj_token.text}'",
                        )

        # Pass 2: Standalone phone pattern match
        for token in tokens:
            norm_text = cls.normalize_digits(token.text)
            phone = cls._extract_phone(norm_text)
            if phone:
                return phone, EvidenceField(
                    value=phone,
                    confidence=min(0.90, token.confidence),
                    source_text=token.text,
                    evidence=f"Pattern-matched Sudanese phone: '{token.text}'",
                )

        return None, EvidenceField(value=None, confidence=0.0)

    @classmethod
    def _extract_phone(cls, text: str) -> Optional[str]:
        clean = re.sub(r"[\s\-\(\)]", "", text)
        for pattern in cls.PHONE_PATTERNS:
            match = re.search(pattern, clean)
            if match:
                digits = match.group(1) if match.groups() else match.group(0)
                if len(digits) == 9 and (digits.startswith("9") or digits.startswith("1")):
                    return f"0{digits}"
                elif len(digits) == 10 and (digits.startswith("09") or digits.startswith("01")):
                    return digits
        return None
