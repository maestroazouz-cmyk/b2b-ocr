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
    Flags future dates relative to current Africa/Khartoum calendar date.
    """

    ARABIC_INDIC_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

    DATE_PATTERNS = [
        # YYYY-MM-DD or YYYY/MM/DD
        (r"\b(20\d{2})[\/\-\.](0[1-9]|1[0-2])[\/\-\.](0[1-9]|[12]\d|3[01])\b", "YMD"),
        # DD-MM-YYYY or DD/MM/YYYY
        (r"\b(0[1-9]|[12]\d|3[01])[\/\-\.](0[1-9]|1[0-2])[\/\-\.](20\d{2})\b", "DMY"),
    ]

    TIME_PATTERNS = [
        # 14:30:15 or 02:30:15 PM
        r"\b([01]?\d|2[0-3]):([0-5]\d):([0-5]\d)(?:\s*(AM|PM|ص|م))?\b",
        # 14:30 or 02:30 PM
        r"\b([01]?\d|2[0-3]):([0-5]\d)(?:\s*(AM|PM|ص|م))?\b",
    ]

    DATE_KEYWORDS = ["التاريخ", "تاريخ العملية", "تاريخ التحويل", "Date", "Transaction Date", "Time"]

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

        # Pass 1: Proximity to date keywords
        for i, token in enumerate(tokens):
            norm_text = cls.normalize_digits(token.text).strip()
            has_kw = any(kw in token.text for kw in cls.DATE_KEYWORDS)

            if has_kw:
                d = cls._extract_date(norm_text)
                t = cls._extract_time(norm_text)
                if d:
                    best_date = d
                    best_time = t
                    evidence = EvidenceField(
                        value=d,
                        confidence=min(0.98, token.confidence),
                        source_text=token.text,
                        evidence=f"Labeled transaction date: '{token.text}'",
                    )
                    break

                for j in range(i + 1, min(i + 3, len(tokens))):
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

        # Pass 2: Standalone date pattern scan
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
                        confidence=min(0.88, token.confidence),
                        source_text=token.text,
                        evidence=f"Pattern-matched calendar date: '{token.text}'",
                    )
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
        for pattern, fmt in cls.DATE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                g = match.groups()
                if fmt == "YMD":
                    year, month, day = g[0], g[1].zfill(2), g[2].zfill(2)
                else:
                    day, month, year = g[0].zfill(2), g[1].zfill(2), g[2]
                return f"{year}-{month}-{day}"
        return None

    @classmethod
    def _extract_time(cls, text: str) -> Optional[str]:
        for pattern in cls.TIME_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
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
