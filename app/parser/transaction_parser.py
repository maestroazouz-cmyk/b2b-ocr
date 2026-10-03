import re
from typing import List, Optional, Tuple
from app.models.schemas import OCRToken, EvidenceField


class TransactionParser:
    """
    Extracts transaction reference / notification number.
    Strict anti-hallucination rules:
    - Requires proximity to explicit transaction labels (رقم المعاملة / الرقم المرجعي / Transaction ID)
      OR valid known prefix formats (e.g. BOK-123456, FT23..., FIB...).
    - Excludes phone numbers, standalone dates, and amounts.
    - If reference number is not clearly visible, returns null (never guesses).
    """

    ARABIC_INDIC_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

    TX_LABELS = [
        "رقم المعاملة",
        "الرقم المرجعي",
        "رقم العملية",
        "رقم الإشعار",
        "رقم القيد",
        "المرجع",
        "Transaction ID",
        "Txn ID",
        "Reference Number",
        "Ref No",
        "Ref #",
        "Transaction Number",
        "FT Number",
    ]

    KNOWN_PREFIX_PATTERNS = [
        r"\b(BOK[-_]?\d{5,12})\b",
        r"\b(FIB[-_]?\d{5,12})\b",
        r"\b(ONB[-_]?\d{5,12})\b",
        r"\b(FT\d{10,16})\b",
        r"\b(TXN[-_]?\d{6,14})\b",
    ]

    @classmethod
    def normalize_digits(cls, text: str) -> str:
        if not text:
            return ""
        return text.translate(cls.ARABIC_INDIC_MAP)

    @classmethod
    def parse(cls, tokens: List[OCRToken]) -> Tuple[Optional[str], EvidenceField]:
        """
        Returns (reference_number, evidence_field)
        """
        # Pass 1: Proximity to transaction labels
        for i, token in enumerate(tokens):
            norm_text = cls.normalize_digits(token.text).strip()
            matched_label = next((lbl for lbl in cls.TX_LABELS if lbl in norm_text), None)

            if matched_label:
                # 1a. Check if reference is in same token after label
                candidate = cls._extract_ref_after_label(norm_text, matched_label)
                if candidate and not cls._is_excluded(candidate):
                    return candidate, EvidenceField(
                        value=candidate,
                        confidence=min(0.98, token.confidence),
                        source_text=token.text,
                        evidence=f"Directly labeled transaction ref: '{token.text}'",
                    )

                # 1b. Check next 3 adjacent tokens
                for j in range(i + 1, min(i + 4, len(tokens))):
                    adj_token = tokens[j]
                    adj_norm = cls.normalize_digits(adj_token.text).strip()
                    cand = cls._clean_ref_candidate(adj_norm)
                    if cand and not cls._is_excluded(cand):
                        conf = min(0.95, (token.confidence + adj_token.confidence) / 2.0)
                        return cand, EvidenceField(
                            value=cand,
                            confidence=conf,
                            source_text=f"{token.text} -> {adj_token.text}",
                            evidence=f"Transaction label '{token.text}' followed by '{adj_token.text}'",
                        )

        # Pass 2: Known banking prefix pattern match
        for token in tokens:
            norm_text = cls.normalize_digits(token.text).strip()
            for pat in cls.KNOWN_PREFIX_PATTERNS:
                match = re.search(pat, norm_text, re.IGNORECASE)
                if match:
                    val = match.group(1).upper()
                    return val, EvidenceField(
                        value=val,
                        confidence=min(0.92, token.confidence),
                        source_text=token.text,
                        evidence=f"Known banking prefix pattern: '{val}'",
                    )

        # Anti-Hallucination: If no explicit label or known prefix exists, do NOT select a random number.
        return None, EvidenceField(value=None, confidence=0.0)

    @classmethod
    def _extract_ref_after_label(cls, text: str, label: str) -> Optional[str]:
        idx = text.find(label)
        if idx != -1:
            remainder = text[idx + len(label) :].strip(" :-\t")
            return cls._clean_ref_candidate(remainder)
        return None

    @classmethod
    def _clean_ref_candidate(cls, text: str) -> Optional[str]:
        clean = re.sub(r"[^\w\-\.]", "", text)
        if len(clean) >= 5 and re.search(r"\d", clean):
            return clean
        return None

    @classmethod
    def _is_excluded(cls, cand: str) -> bool:
        # Exclude dates (e.g. 2024-10-15 or 15/10/2024)
        if re.match(r"^\d{4}[\-\/\.]\d{2}[\-\/\.]\d{2}$", cand) or re.match(r"^\d{2}[\-\/\.]\d{2}[\-\/\.]\d{4}$", cand):
            return True
        # Exclude phone numbers (09... or 01...)
        if (cand.startswith("09") or cand.startswith("01")) and len(cand) == 10 and cand.isdigit():
            return True
        # Exclude pure short numbers (<5 digits)
        if cand.isdigit() and len(cand) < 5:
            return True
        return False
