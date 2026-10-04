import re
from datetime import datetime
from typing import List, Optional, Tuple
from app.models.schemas import OCRToken, EvidenceField, ValidationMetadata

try:
    import pytz
    KHARTOUM_TZ = pytz.timezone("Africa/Khartoum")
except ImportError:
    KHARTOUM_TZ = None


class DateParser:
    """
    Extracts transaction date and time with strict Africa/Khartoum timezone grounding.
    Supports ISO, European, and Named Month date formats (e.g. 03-Oct-2026, 02-Oc-2026, 03/10/2026).
    Handles localized OCR substitutions (e.g. O3-0ct-2026, 02-Oc-2026 -> 2026-10-02).
    Flags future dates relative to current Africa/Khartoum calendar date.
    """

    ARABIC_INDIC_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

    MONTH_MAP = {
        # January
        "jan": "01", "january": "01", "ja": "01", "يناير": "01",
        # February
        "feb": "02", "february": "02", "fe": "02", "فبراير": "02",
        # March
        "mar": "03", "march": "03", "mr": "03", "مارس": "03",
        # April
        "apr": "04", "april": "04", "ap": "04", "أبريل": "04", "ابريل": "04",
        # May
        "may": "05", "my": "05", "مايو": "05",
        # June
        "jun": "06", "june": "06", "jn": "06", "يونيو": "06",
        # July
        "jul": "07", "july": "07", "jl": "07", "يوليو": "07",
        # August
        "aug": "08", "august": "08", "au": "08", "أغسطس": "08", "اغسطس": "08",
        # September
        "sep": "09", "september": "09", "se": "09", "5ep": "09", "سبتمبر": "09",
        # October (support full, 3-char, 2-char, and OCR zero substitutions)
        "oct": "10", "october": "10", "oc": "10", "0ct": "10", "0c": "10", "ocl": "10", "0ctober": "10", "أكتوبر": "10", "اكتوبر": "10",
        # November
        "nov": "11", "november": "11", "no": "11", "نوفمبر": "11",
        # December
        "dec": "12", "december": "12", "de": "12", "ديسمبر": "12",
    }

    DATE_PATTERNS = [
        # Named month: DD-Mon-YYYY (e.g. 03-Oct-2026, 02-Oc-2026, 03/Oct/2026, O2-0ct-2026, 02-Oct-26)
        (r"\b(0?[1-9]|[12]\d|3[01])[\/\-\s]([A-Za-z0-9]{2,9}|[\u0600-\u06FF]{2,9})[\/\-\s](20\d{2}|\d{2})\b", "TEXT_DMY"),
        # YYYY-MM-DD or YYYY/MM/DD
        (r"\b(20\d{2})[\/\-\.](0?[1-9]|1[0-2])[\/\-\.](0?[1-9]|[12]\d|3[01])\b", "YMD"),
        # DD-MM-YYYY or DD/MM/YYYY
        (r"\b(0?[1-9]|[12]\d|3[01])[\/\-\.](0?[1-9]|1[0-2])[\/\-\.](20\d{2}|\d{2})\b", "DMY"),
    ]

    TIME_PATTERNS = [
        # 10:52:07 or 19:36:41 or 14:30:15 or 02:30:15 PM
        r"\b([01]?\d|2[0-3]):([0-5]\d):([0-5]\d)(?:\s*(AM|PM|ص|م))?\b",
        # 14:30 or 02:30 PM
        r"\b([01]?\d|2[0-3]):([0-5]\d)(?:\s*(AM|PM|ص|م))?\b",
    ]

    DATE_KEYWORDS = [
        "التاريخ",
        "التاريخ والزمن",
        "تاريخ العملية",
        "تاريخ التحويل",
        "Date",
        "Transaction Date",
        "Time",
        "Date & Time",
    ]

    @classmethod
    def get_current_khartoum_date(cls) -> str:
        if KHARTOUM_TZ:
            now = datetime.now(KHARTOUM_TZ)
        else:
            now = datetime.utcnow()
        return now.strftime("%Y-%m-%d")

    @classmethod
    def normalize_digits(cls, text: str) -> str:
        if not text:
            return ""
        return text.translate(cls.ARABIC_INDIC_MAP)

    @classmethod
    def parse(
        cls, tokens: List[OCRToken]
    ) -> Tuple[Optional[str], Optional[str], EvidenceField, ValidationMetadata]:
        """
        Returns (date_str_iso, time_str, evidence_field, validation_metadata)
        """
        khartoum_today = cls.get_current_khartoum_date()
        best_date: Optional[str] = None
        best_time: Optional[str] = None
        evidence = EvidenceField(value=None, confidence=0.0)

        # Pass 1: Proximity to date keywords or direct inspection
        for i, token in enumerate(tokens):
            norm_text = cls.normalize_digits(token.text).strip()
            has_kw = any(kw in token.text for kw in cls.DATE_KEYWORDS)

            if has_kw:
                d = cls._extract_date(norm_text)
                t = cls._extract_time(norm_text)
                if d:
                    best_date = d
                    best_time = t or best_time
                    evidence = EvidenceField(
                        value=d,
                        confidence=min(0.98, token.confidence),
                        source_text=token.text,
                        evidence=f"Labeled transaction date: '{token.text}'",
                    )
                    break

                # Scan adjacent tokens in vertical proximity
                for j in range(i + 1, min(i + 4, len(tokens))):
                    adj = tokens[j]
                    adj_norm = cls.normalize_digits(adj.text).strip()
                    d = cls._extract_date(adj_norm)
                    t = cls._extract_time(adj_norm)
                    if d:
                        best_date = d
                        best_time = t or best_time
                        conf = min(0.95, (token.confidence + adj.confidence) / 2.0)
                        evidence = EvidenceField(
                            value=d,
                            confidence=conf,
                            source_text=f"{token.text} -> {adj.text}",
                            evidence=f"Date keyword '{token.text}' followed by '{adj.text}'",
                        )
                        break
                if best_date:
                    break

        # Pass 2: Standalone date scan across all tokens
        if not best_date:
            for token in tokens:
                norm_text = cls.normalize_digits(token.text).strip()
                d = cls._extract_date(norm_text)
                t = cls._extract_time(norm_text)
                if d:
                    best_date = d
                    best_time = t or best_time
                    evidence = EvidenceField(
                        value=d,
                        confidence=min(0.90, token.confidence),
                        source_text=token.text,
                        evidence=f"Pattern-matched calendar date: '{token.text}'",
                    )
                    break

        # If time is still missing, scan for standalone time pattern
        if not best_time:
            for token in tokens:
                norm_text = cls.normalize_digits(token.text).strip()
                t = cls._extract_time(norm_text)
                if t:
                    best_time = t
                    break

        # Timezone validation
        is_future = False
        is_today = False
        notes = ""

        if best_date:
            if best_date > khartoum_today:
                is_future = True
                notes = f"Transaction date ({best_date}) is in the future relative to Africa/Khartoum ({khartoum_today})."
            elif best_date == khartoum_today:
                is_today = True
                notes = "Transaction executed today in Africa/Khartoum (Valid)."
            else:
                notes = "Historical transaction date in Africa/Khartoum (Valid)."

        val_meta = ValidationMetadata(
            timezone="Africa/Khartoum",
            current_date_in_khartoum=khartoum_today,
            is_future_date=is_future,
            is_today_date=is_today,
            date_interpretation_notes=notes,
        )

        return best_date, best_time, evidence, val_meta

    @classmethod
    def _extract_date(cls, text: str) -> Optional[str]:
        if not text:
            return None

        s = text.strip()
        # Conservative OCR substitutions inside date candidate
        s = re.sub(r"\b[Oo](\d)", r"0\1", s)
        s = re.sub(r"(\-)[Oo](\d)", r"\g<1>0\g<2>", s)
        s = re.sub(r"(\/)[Oo](\d)", r"\g<1>0\g<2>", s)
        s = re.sub(r"\b[Oo]([A-Za-z])", r"0\1", s)

        # Normalize common OCR month representations
        s = s.replace("0ct", "Oct").replace("0CT", "Oct").replace("oCT", "Oct").replace("oct", "Oct")
        s = re.sub(r"\b0c\b", "Oct", s, flags=re.IGNORECASE)
        s = re.sub(r"\boc\b", "Oct", s, flags=re.IGNORECASE)
        s = s.replace("5ep", "Sep").replace("sep", "Sep")

        for pattern, fmt in cls.DATE_PATTERNS:
            match = re.search(pattern, s, re.IGNORECASE)
            if match:
                g = match.groups()
                if fmt == "TEXT_DMY":
                    day = g[0].zfill(2)
                    month_key = g[1].lower()
                    year_raw = g[2]
                    year = f"20{year_raw}" if len(year_raw) == 2 else year_raw
                    month_num = cls.MONTH_MAP.get(month_key)
                    if not month_num:
                        # Fallback for prefixes like 'oc' or '0c'
                        for k, v in cls.MONTH_MAP.items():
                            if month_key.startswith(k) or k.startswith(month_key):
                                month_num = v
                                break
                    if month_num:
                        return f"{year}-{month_num}-{day}"
                elif fmt == "YMD":
                    year, month, day = g[0], g[1].zfill(2), g[2].zfill(2)
                    return f"{year}-{month}-{day}"
                elif fmt == "DMY":
                    day = g[0].zfill(2)
                    month = g[1].zfill(2)
                    year_raw = g[2]
                    year = f"20{year_raw}" if len(year_raw) == 2 else year_raw
                    return f"{year}-{month}-{day}"
        return None

    @classmethod
    def _extract_time(cls, text: str) -> Optional[str]:
        if not text:
            return None
        s = re.sub(r"\b[lI](\d):", r"1\1:", text.strip())

        for pattern in cls.TIME_PATTERNS:
            match = re.search(pattern, s, re.IGNORECASE)
            if match:
                g = match.groups()
                hour = int(g[0])
                minute = g[1].zfill(2)
                second = g[2].zfill(2) if len(g) > 2 and g[2] else "00"
                ampm = g[3] if len(g) > 3 and g[3] else None

                if ampm:
                    ampm_clean = ampm.upper()
                    if (ampm_clean in ["PM", "م"]) and hour < 12:
                        hour += 12
                    elif (ampm_clean in ["AM", "ص"]) and hour == 12:
                        hour = 0

                return f"{hour:02d}:{minute}:{second}"
        return None
