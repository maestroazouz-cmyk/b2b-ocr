import re
from typing import List, Optional, Tuple
from app.models.schemas import OCRToken, EvidenceField


class AmountParser:
    """
    Extracts payment amount and currency from Sudanese bank vouchers.
    Converts Arabic-Indic numerals, removes comma/space separators, excludes phone numbers and dates.
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

        # Detect currency across all tokens first
        full_text = " ".join(t.text for t in tokens)
        for curr_code, symbols in cls.CURRENCY_INDICATORS.items():
            for sym in symbols:
                if re.search(r"\b" + re.escape(sym) + r"\b", full_text, re.IGNORECASE):
                    currency = curr_code
                    break

        # Pass 1: Look for Amount label proximity
        for i, token in enumerate(tokens):
            norm_text = token.text.strip()
            is_amount_label = any(kw in norm_text for kw in cls.AMOUNT_KEYWORDS)

            if is_amount_label:
                # 1a. Check inside same token
                amt, fmt = cls._extract_amount_from_string(norm_text)
                if amt is not None and not cls._is_phone_or_date(amt, norm_text):
                    best_amount = amt
                    best_formatted = f"{fmt} {currency}"
                    evidence = EvidenceField(
                        value=amt,
                        confidence=min(0.98, max(0.85, token.confidence)),
                        source_text=token.text,
                        evidence=f"Labeled amount: '{token.text}'",
                    )
                    return best_amount, best_formatted, currency, evidence

                # 1b. Check next 3 adjacent tokens
                for j in range(i + 1, min(i + 4, len(tokens))):
                    adj_token = tokens[j]
                    adj_norm = cls.normalize_digits(adj_token.text)
                    amt, fmt = cls._extract_amount_from_string(adj_norm)
                    if amt is not None and not cls._is_phone_or_date(amt, adj_norm):
                        best_amount = amt
                        best_formatted = f"{fmt} {currency}"
                        conf = min(0.96, (token.confidence + adj_token.confidence) / 2.0)
                        evidence = EvidenceField(
                            value=amt,
                            confidence=conf,
                            source_text=f"{token.text} -> {adj_token.text}",
                            evidence=f"Amount label '{token.text}' followed by '{adj_token.text}'",
                        )
                        return best_amount, best_formatted, currency, evidence

        # Pass 2: Standalone numbers with decimal patterns (e.g. 150,000.00 or 150000.00)
        candidates = []
        for token in tokens:
            norm_text = cls.normalize_digits(token.text)
            amt, fmt = cls._extract_amount_from_string(norm_text)
            if amt is not None and not cls._is_phone_or_date(amt, norm_text):
                # Filter out suspicious year numbers or short IDs
                if amt > 10.0 and amt != 2024 and amt != 2025 and amt != 2026:
                    candidates.append((amt, fmt, token))

        if candidates:
            # Pick candidate with highest decimal precision or largest realistic transfer
            amt, fmt, token = candidates[0]
            best_amount = amt
            best_formatted = f"{fmt} {currency}"
            evidence = EvidenceField(
                value=amt,
                confidence=min(0.80, token.confidence * 0.9),
                source_text=token.text,
                evidence=f"Pattern-matched number: '{token.text}'",
            )

        return best_amount, best_formatted, currency, evidence

    @classmethod
    def _extract_amount_from_string(cls, text: str) -> Tuple[Optional[float], Optional[str]]:
        clean = cls.normalize_digits(text)
        # Matches patterns like: 150,000.00 | 150 000 | 150000.50 | 150000
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
    def _is_phone_or_date(cls, val: float, raw_str: str) -> bool:
        clean = cls.normalize_digits(raw_str).replace(" ", "").replace("-", "")
        # Check if starts with Sudanese phone prefixes
        if clean.startswith("09") or clean.startswith("01") or clean.startswith("249"):
            return True
        # Check if looks like a date (2024-2027)
        if 2020 <= val <= 2030:
            return True
        return False
