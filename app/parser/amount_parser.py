import re
from typing import List, Optional, Tuple
from app.models.schemas import OCRToken, EvidenceField


class AmountParser:
    """
    Extracts payment amount and currency from Sudanese bank vouchers.
    Enforces strict anti-hallucination guardrails:
    - Amount must come strictly from the field/tokens associated with the label 'المبلغ'
      or directly adjacent to currency symbols (SDG / ج.س).
    - NEVER treats transaction IDs, account numbers, dates, or phone numbers as amounts.
    - Accurately normalizes reversed OCR digit patterns (e.g. '000.00,35' -> 35,000.00).
    - If amount is ambiguous, returns None (review_required = True).
    """

    ARABIC_INDIC_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

    AMOUNT_KEYWORDS = [
        "المبلغ",
        "مبلغ",
        "المبلغ المحول",
        "القيمة",
        "المجموع",
        "Amount",
        "Total",
        "Transfer Amount",
        "Paid",
        "غلبملا",
        "غلبا",
        "غلUnit",
    ]

    CURRENCY_INDICATORS = {
        "SDG": ["ج.س", "جنيه", "جنيه سوداني", "SDG", "SD", "LS", "ج"],
        "USD": ["USD", "$", "دولار"],
        "EUR": ["EUR", "€", "يورو"],
        "SAR": ["SAR", "ريال", "ريال سعودي"],
        "AED": ["AED", "درهم", "درهم إماراتي"],
    }

    @classmethod
    def normalize_digits(cls, text: str) -> str:
        if not text:
            return ""
        return text.translate(cls.ARABIC_INDIC_MAP)

    @classmethod
    def parse(cls, tokens: List[OCRToken]) -> Tuple[Optional[float], Optional[str], str, EvidenceField]:
        """
        Returns (amount_val, formatted_amount, currency, evidence_field)
        """
        best_amount: Optional[float] = None
        best_formatted: Optional[str] = None
        currency: str = "SDG"
        evidence = EvidenceField(value=None, confidence=0.0)

        # Detect currency across tokens
        full_text = " ".join(t.text for t in tokens)
        for curr_code, symbols in cls.CURRENCY_INDICATORS.items():
            for sym in symbols:
                if re.search(r"\b" + re.escape(sym) + r"\b", full_text, re.IGNORECASE):
                    currency = curr_code
                    break

        # Pass 1: Look for explicit Amount label proximity
        for i, token in enumerate(tokens):
            norm_text = token.text.strip()
            is_amount_label = any(kw in norm_text for kw in cls.AMOUNT_KEYWORDS)

            if is_amount_label:
                # 1a. Check inside same token
                amt, fmt = cls._extract_amount_from_string(norm_text)
                if amt is not None and not cls._is_non_amount_entity(amt, norm_text):
                    best_amount = amt
                    best_formatted = f"{fmt} {currency}"
                    evidence = EvidenceField(
                        value=amt,
                        confidence=min(0.98, max(0.85, token.confidence)),
                        source_text=token.text,
                        evidence=f"Extracted from value region associated with label 'المبلغ': '{token.text}'",
                    )
                    return best_amount, best_formatted, currency, evidence

                # 1b. Check next adjacent tokens (by line or proximity)
                for j in range(i + 1, min(i + 4, len(tokens))):
                    adj_token = tokens[j]
                    adj_norm = cls.normalize_digits(adj_token.text)
                    amt, fmt = cls._extract_amount_from_string(adj_norm)
                    if amt is not None and not cls._is_non_amount_entity(amt, adj_norm):
                        best_amount = amt
                        best_formatted = f"{fmt} {currency}"
                        conf = min(0.96, (token.confidence + adj_token.confidence) / 2.0)
                        evidence = EvidenceField(
                            value=amt,
                            confidence=conf,
                            source_text=f"{token.text} -> {adj_token.text}",
                            evidence=f"Extracted from value region associated with label 'المبلغ': '{adj_token.text}'",
                        )
                        return best_amount, best_formatted, currency, evidence

        # Pass 2: Explicit currency symbol adjacency (e.g. '35,000.00 SDG' or '150,000 ج.س')
        for token in tokens:
            norm_text = cls.normalize_digits(token.text).strip()
            has_currency_marker = any(
                sym in norm_text for sym in ["SDG", "ج.س", "جنيه", "USD", "SAR", "AED"]
            )
            if has_currency_marker:
                amt, fmt = cls._extract_amount_from_string(norm_text)
                if amt is not None and not cls._is_non_amount_entity(amt, norm_text):
                    best_amount = amt
                    best_formatted = f"{fmt} {currency}"
                    evidence = EvidenceField(
                        value=amt,
                        confidence=min(0.92, token.confidence),
                        source_text=token.text,
                        evidence=f"Currency-bounded amount token: '{token.text}'",
                    )
                    return best_amount, best_formatted, currency, evidence

        # Anti-Hallucination: Never guess arbitrary numbers from the page!
        return None, None, currency, EvidenceField(value=None, confidence=0.0)

    @classmethod
    def _extract_amount_from_string(cls, text: str) -> Tuple[Optional[float], Optional[str]]:
        clean = cls.normalize_digits(text).strip()

        # 1. Check reversed decimal notation: e.g. "000.00,35" or "000.00,50" -> 35,000.00 / 50,000.00
        reversed_match = re.search(r"(\d{2,3})\.(\d{2}),(\d{1,3})", clean)
        if reversed_match:
            g = reversed_match.groups()
            num_str = f"{g[2]}{g[0]}.{g[1]}"
            try:
                val = float(num_str)
                if val > 0:
                    return val, f"{val:,.2f}"
            except ValueError:
                pass

        # 2. Check reversed pattern without comma: e.g. "00.000,35"
        reversed_match_2 = re.search(r"(\d{2})\.(\d{3}),(\d{1,3})", clean)
        if reversed_match_2:
            g = reversed_match_2.groups()
            num_str = f"{g[2]}{g[1]}.{g[0]}"
            try:
                val = float(num_str)
                if val > 0:
                    return val, f"{val:,.2f}"
            except ValueError:
                pass

        # 3. Standard patterns: 35,000.00 | 35000.00 | 35,000 | 35000 | 150,000.50
        matches = re.findall(r"(?:\d{1,3}(?:[,\s]\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)", clean)
        for m in matches:
            num_str = m.replace(",", "").replace(" ", "")
            try:
                val = float(num_str)
                if val > 0:
                    formatted = f"{val:,.2f}"
                    return val, formatted
            except ValueError:
                continue

        return None, None

    @classmethod
    def _is_non_amount_entity(cls, val: float, raw_str: str) -> bool:
        """
        Rejects long reference IDs (e.g. 20265282625), accounts (0373 1204 4436 0001),
        dates (2026), and phone numbers (0912345678).
        """
        digits_only = re.sub(r"[^\d]", "", cls.normalize_digits(raw_str))

        # Long integer >= 8 digits with no cents (like transaction ID 20265282625 or 2021531441)
        if val.is_integer() and len(str(int(val))) >= 8:
            # If not explicitly formatted as currency with both commas and dots like 10,000,000.00
            if not ("," in raw_str and "." in raw_str):
                return True

        # Sudanese phone numbers
        if digits_only.startswith("09") or digits_only.startswith("01") or digits_only.startswith("249"):
            if len(digits_only) in [10, 12]:
                return True

        # Year check (2020-2030)
        if 2020 <= val <= 2030 and val.is_integer() and len(digits_only) == 4:
            return True

        return False
