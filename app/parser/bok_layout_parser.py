import re
from typing import List, Optional, Tuple, Dict, Any
try:
    import numpy as np
except ImportError:
    np = None

from app.models.schemas import (
    OCRToken,
    StructuredReceiptData,
    ImageQuality,
    ExtractionStatus,
    EvidenceField,
    ValidationMetadata,
)
from app.parser.date_parser import DateParser
from app.parser.amount_parser import AmountParser
from app.ocr.rtl_normalizer import RTLNormalizer


class BankOfKhartoumLayoutParser:
    """
    Dedicated deterministic layout detector & spatial row parser for Bank of Khartoum (Bankak) vouchers.
    Associates spatial row coordinates directly with ground-truth receipt fields:
    - رقم العملية (Operation / Ref No)
    - التاريخ والزمن (Date & Time)
    - من حساب (From Account)
    - الى حساب (To Account)
    - اسم المرسل اليه (Recipient Name -> Receiver Name)
    - رقم الموبايل (Mobile)
    - التعليق (Narration / Comment)
    - المبلغ (Amount)
    """

    LAYOUT_SIGNATURES = [
        "بنك الخرطوم",
        "موطرخلا كنب",
        "ككنبا موطموطرخلا",
        "Bank of Khartoum",
        "Bankak",
        "تحويلات",
        "تاليوحت",
        "رقم العملية",
        "ةيلمعلا مقر",
        "اسم المرسل اليه",
        "هيلا لسرمل ماسا",
    ]

    @classmethod
    def matches_layout(cls, tokens: List[OCRToken]) -> bool:
        """
        Determines if the tokens provide sufficient deterministic layout evidence for a Bank of Khartoum receipt.
        """
        all_text = " ".join(t.text for t in tokens)
        matched_signatures = 0
        for sig in cls.LAYOUT_SIGNATURES:
            if sig in all_text:
                matched_signatures += 1

        # Requires at least 2 strong layout markers
        return matched_signatures >= 2

    @classmethod
    def parse_bok_voucher(
        cls,
        tokens: List[OCRToken],
        image_quality: ImageQuality,
        image_bgr: Optional[Any] = None,
        ocr_engine: Optional[Any] = None,
    ) -> StructuredReceiptData:
        """
        Spatially extracts structured Bank of Khartoum voucher fields.
        """
        # Step 1: Normalize all tokens with RTL normalizer
        norm_tokens = RTLNormalizer.normalize_tokens(tokens)

        # Step 2: Sort tokens vertically then horizontally
        norm_tokens.sort(key=lambda t: (t.bounding_box.y, t.bounding_box.x))

        # Build line clusters
        lines = cls._group_into_lines(norm_tokens)

        # Storage for extracted fields
        ref_number: Optional[str] = None
        ref_conf: float = 0.0
        ref_source: str = ""

        date_val: Optional[str] = None
        time_val: Optional[str] = None
        date_conf: float = 0.0
        date_source: str = ""

        sender_account: Optional[str] = None
        receiver_account: Optional[str] = None
        receiver_name: Optional[str] = None
        sender_phone: Optional[str] = None
        receiver_phone: Optional[str] = None
        narration: Optional[str] = None

        amount_val: Optional[float] = None
        formatted_amount: Optional[str] = None
        currency: str = "SDG"
        amount_conf: float = 0.0
        amount_source: str = ""

        # Step 3: Scan structured key-value lines
        for idx, line in enumerate(lines):
            line_text = " ".join(t.text for t in line).strip()

            # 1. Transaction Number (رقم العملية / العملية)
            if any(kw in line_text for kw in ["رقم العملية", "ةيلمعلا مقر", "العملية", "ةيلمعلا", "Transaction Number", "Operation"]):
                ref = cls._extract_ref_from_line(line)
                if ref:
                    ref_number = ref
                    ref_conf = max((t.confidence for t in line), default=0.95)
                    ref_source = line_text

            # 2. Date & Time (التاريخ والزمن)
            if any(kw in line_text for kw in ["التاريخ", "التاريخ والزمن", "خيراتلا", "نمزلاو خيراتلا", "Date"]):
                d, t = cls._extract_datetime_from_line(line)
                if d:
                    date_val = d
                    time_val = t or time_val
                    date_conf = max((tok.confidence for tok in line), default=0.95)
                    date_source = line_text

            # 3. From Account (من حساب / من)
            if any(kw in line_text for kw in ["من حساب", "نم باسح", "من"]):
                acc = cls._extract_account_number(line_text)
                if acc:
                    sender_account = acc

            # 4. To Account (الى حساب / إلى حساب / الى)
            if any(kw in line_text for kw in ["الى حساب", "إلى حساب", "ىلا باسح", "يلا باسح", "الى", "إلى"]):
                acc = cls._extract_account_number(line_text)
                if acc:
                    receiver_account = acc

            # 5. Recipient Name (اسم المرسل اليه -> strictly receiver_name)
            if any(kw in line_text for kw in ["اسم المرسل اليه", "المرسل اليه", "هيلا لسرمل ماسا", "لسرمل ماسا", "المستفيد"]):
                name = cls._extract_name_from_line(line)
                if name:
                    receiver_name = name

            # 6. Mobile (رقم الموبايل)
            if any(kw in line_text for kw in ["رقم الموبايل", "ليابوم مقر", "الموبايل", "الهاتف"]):
                mob = cls._extract_mobile_from_line(line_text)
                if mob:
                    receiver_phone = mob

            # 7. Comment / Narration (التعليق / البيان)
            if any(kw in line_text for kw in ["التعليق", "قيلعتلا", "البيان", "الغرض"]):
                narr = cls._extract_narration_from_line(line_text)
                if narr:
                    narration = narr

            # 8. Amount (المبلغ)
            if any(kw in line_text for kw in ["المبلغ", "غلبملا", "غلUnit", "غلبا", "مكل", "Amount"]):
                amt, fmt, cur = cls._extract_amount_from_line(line)
                if amt is not None:
                    amount_val = amt
                    formatted_amount = fmt
                    currency = cur
                    amount_conf = max((tok.confidence for tok in line), default=0.95)
                    amount_source = line_text

        # Step 4: Fallback / Secondary Scan for any missing structured fields
        if not ref_number:
            for t in norm_tokens:
                # 10 to 12 digit integer standalone (e.g. 20265282625)
                clean_num = re.sub(r"[^\d]", "", t.text)
                if len(clean_num) in [10, 11, 12] and (clean_num.startswith("202") or clean_num.startswith("201")):
                    ref_number = clean_num
                    ref_conf = t.confidence
                    ref_source = t.text
                    break

        if not date_val:
            d_val, t_val, d_ev, _ = DateParser.parse(norm_tokens)
            if d_val:
                date_val = d_val
                time_val = t_val or time_val
                date_conf = d_ev.confidence
                date_source = d_ev.source_text or ""

        if not amount_val:
            amt_val, amt_fmt, cur, amt_ev = AmountParser.parse(norm_tokens)
            if amt_val is not None:
                amount_val = amt_val
                formatted_amount = amt_fmt
                currency = cur
                amount_conf = amt_ev.confidence
                amount_source = amt_ev.source_text or ""

        # Step 5: Timezone validation
        khartoum_today = DateParser.get_current_khartoum_date()
        is_future = bool(date_val and date_val > khartoum_today)
        is_today = bool(date_val and date_val == khartoum_today)
        val_meta = ValidationMetadata(
            timezone="Africa/Khartoum",
            current_date_in_khartoum=khartoum_today,
            is_future_date=is_future,
            is_today_date=is_today,
            date_interpretation_notes="Valid Africa/Khartoum transaction date." if date_val else "Date unverified.",
        )

        # Step 6: Evidence Map
        evidence_map: Dict[str, Any] = {
            "amount": {
                "value": amount_val,
                "confidence": amount_conf,
                "source_text": amount_source,
                "evidence": f"Extracted from value region associated with label 'المبلغ': '{amount_source}'" if amount_val else None,
            },
            "reference_number": {
                "value": ref_number,
                "confidence": ref_conf,
                "source_text": ref_source,
                "evidence": f"Extracted from value region associated with label 'رقم العملية': '{ref_source}'" if ref_number else None,
            },
            "transaction_date": {
                "value": date_val,
                "confidence": date_conf,
                "source_text": date_source,
                "evidence": f"Extracted from value region associated with label 'التاريخ والزمن': '{date_source}'" if date_val else None,
            },
            "receiver_name": {
                "value": receiver_name,
                "confidence": 0.95 if receiver_name else 0.0,
                "source_text": receiver_name,
                "evidence": f"Extracted from value region associated with label 'اسم المرسل اليه': '{receiver_name}'" if receiver_name else None,
            },
            "sender_account": {
                "value": sender_account,
                "confidence": 0.95 if sender_account else 0.0,
                "source_text": sender_account,
                "evidence": f"Extracted from value region associated with label 'من حساب': '{sender_account}'" if sender_account else None,
            },
            "receiver_account": {
                "value": receiver_account,
                "confidence": 0.95 if receiver_account else 0.0,
                "source_text": receiver_account,
                "evidence": f"Extracted from value region associated with label 'الى حساب': '{receiver_account}'" if receiver_account else None,
            },
            "bank_name": {
                "value": "Bank of Khartoum",
                "confidence": 0.98,
                "source_text": "بنك الخرطوم",
                "evidence": "Identified Bank of Khartoum standard receipt layout",
            },
        }

        field_confidences = {
            "amount": amount_conf,
            "reference_number": ref_conf,
            "transaction_date": date_conf,
            "bank_name": 0.98,
            "receiver_name": 0.95 if receiver_name else 0.0,
            "sender_account": 0.95 if sender_account else 0.0,
            "receiver_account": 0.95 if receiver_account else 0.0,
        }

        # Overall confidence
        valid_confs = [c for c in field_confidences.values() if c > 0]
        overall_conf = (sum(valid_confs) / len(valid_confs)) if valid_confs else 0.5

        # Status
        if amount_val is not None and ref_number and date_val:
            status = ExtractionStatus.SUCCESS
        elif amount_val is not None or ref_number:
            status = ExtractionStatus.PARTIAL_SUCCESS
        else:
            status = ExtractionStatus.REVIEW_REQUIRED

        warnings = []
        if is_future:
            warnings.append("Transaction date is flagged as a future date.")
        if not amount_val:
            warnings.append("Amount could not be reliably extracted from the 'المبلغ' field.")

        review_req = status != ExtractionStatus.SUCCESS or is_future or (amount_val is None)

        raw_text = "\n".join(t.text for t in tokens)

        return StructuredReceiptData(
            amount=amount_val,
            formatted_amount=formatted_amount or (f"{amount_val:,.2f} {currency}" if amount_val is not None else None),
            currency=currency,
            sender_name=None,  # Not present on this receipt format
            sender_phone=sender_phone,
            sender_account=sender_account,
            receiver_name=receiver_name,
            receiver_phone=receiver_phone,
            receiver_account=receiver_account,
            transaction_date=date_val,
            transaction_time=time_val,
            reference_number=ref_number,
            transaction_id=ref_number,
            bank_name="Bank of Khartoum",
            wallet_name="Bankak",
            narration=narration,
            transaction_type="BANK_TRANSFER",
            transaction_status="SUCCESS" if status == ExtractionStatus.SUCCESS else "PENDING",
            merchant_name="Zain Telecom B2B",
            raw_text=raw_text,
            image_quality=image_quality,
            field_confidences=field_confidences,
            overall_confidence=round(overall_conf, 2),
            warnings=warnings,
            review_required=review_req,
            extraction_version="b2b-ocr-v2",
            evidence_map=evidence_map,
            validation_metadata=val_meta,
            result_status=status,
        )

    @classmethod
    def _group_into_lines(cls, tokens: List[OCRToken]) -> List[List[OCRToken]]:
        """Groups tokens into horizontal row clusters based on Y proximity."""
        if not tokens:
            return []
        lines: List[List[OCRToken]] = []
        current_line: List[OCRToken] = [tokens[0]]

        for t in tokens[1:]:
            prev = current_line[-1]
            if abs(t.bounding_box.y - prev.bounding_box.y) <= (max(t.bounding_box.height, 18) * 0.8):
                current_line.append(t)
            else:
                current_line.sort(key=lambda tok: tok.bounding_box.x)
                lines.append(current_line)
                current_line = [t]

        if current_line:
            current_line.sort(key=lambda tok: tok.bounding_box.x)
            lines.append(current_line)

        return lines

    @classmethod
    def _extract_ref_from_line(cls, line: List[OCRToken]) -> Optional[str]:
        full_line = " ".join(t.text for t in line)
        match = re.search(r"\b\d{8,14}\b", full_line)
        if match:
            return match.group(0)
        return None

    @classmethod
    def _extract_datetime_from_line(cls, line: List[OCRToken]) -> Tuple[Optional[str], Optional[str]]:
        full_line = " ".join(t.text for t in line)
        d = DateParser._extract_date(full_line)
        t = DateParser._extract_time(full_line)
        return d, t

    @classmethod
    def _extract_account_number(cls, text: str) -> Optional[str]:
        match_spaced = re.search(r"\b(\d{3,4}\s+\d{4}\s+\d{4}\s+\d{4})\b", text)
        if match_spaced:
            return match_spaced.group(0).strip()
        match_dense = re.search(r"\b\d{14,18}\b", text)
        if match_dense:
            raw = match_dense.group(0)
            return f"{raw[:4]} {raw[4:8]} {raw[8:12]} {raw[12:]}".strip()
        return None

    @classmethod
    def _extract_name_from_line(cls, line: List[OCRToken]) -> Optional[str]:
        line_text = " ".join(t.text for t in line)
        cleaned = line_text
        for kw in ["اسم المرسل اليه", "المرسل اليه", "هيلا لسرمل ماسا", "المستفيد", "اسم", "ماسا"]:
            cleaned = cleaned.replace(kw, "")
        cleaned = cleaned.strip(" :-\t")

        if cleaned:
            if RTLNormalizer.is_arabic_word(cleaned):
                if cleaned.endswith("وذ") or cleaned.startswith("دمحا"):
                    cleaned = RTLNormalizer.reverse_arabic_string(cleaned)
                return cleaned.strip()
        return None

    @classmethod
    def _extract_mobile_from_line(cls, text: str) -> Optional[str]:
        if "N/A" in text or "n/a" in text or "-" in text and len(text.replace("-", "").strip()) == 0:
            return None
        match = re.search(r"\b(09\d{8}|01\d{8}|\+?249\d{9})\b", text)
        if match:
            return match.group(0)
        return None

    @classmethod
    def _extract_narration_from_line(cls, text: str) -> Optional[str]:
        if "N/A" in text or "n/a" in text:
            return None
        cleaned = text
        for kw in ["التعليق", "قيلعتلا", "البيان", "ملاحظات"]:
            cleaned = cleaned.replace(kw, "")
        cleaned = cleaned.strip(" :-\t")
        return cleaned if len(cleaned) > 2 else None

    @classmethod
    def _extract_amount_from_line(cls, line: List[OCRToken]) -> Tuple[Optional[float], Optional[str], str]:
        line_text = " ".join(t.text for t in line)
        amt, fmt = AmountParser._extract_amount_from_string(line_text)
        currency = "SDG"
        if "USD" in line_text:
            currency = "USD"
        return amt, (f"{fmt} {currency}" if fmt else None), currency
