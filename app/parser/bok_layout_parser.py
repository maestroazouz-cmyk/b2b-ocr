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
from app.parser.arabic_name_processor import ArabicNameProcessor
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
        "من حساب",
        "الى حساب",
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
        receiver_name_raw: Optional[str] = None
        receiver_name_conf: float = 0.0
        receiver_name_val_meta: Dict[str, Any] = {}
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
                acc = cls._extract_bok_account(line)
                if acc:
                    sender_account = acc

            # 4. To Account (الى حساب / إلى حساب / الى / حساب لا / ىلا باسح)
            if any(kw in line_text for kw in ["الى حساب", "إلى حساب", "ىلا باسح", "يلا باسح", "حساب لا", "باسح لا", "الى", "إلى"]):
                acc = cls._extract_bok_account(line)
                if acc:
                    receiver_account = acc

            # 5. Recipient Name (اسم المرسل اليه -> strictly receiver_name)
            if any(kw in line_text for kw in ["اسم المرسل اليه", "المرسل اليه", "هيلا لسرمل ماسا", "لسرمل ماسا", "المستفيد", "المرسل", "اليه", "السرملا", "إسم", "مسأ"]):
                norm_n, raw_n, n_conf, n_val = ArabicNameProcessor.extract_name_from_tokens(line, "receiver_name")
                if norm_n:
                    receiver_name = norm_n
                    receiver_name_raw = raw_n
                    receiver_name_conf = n_conf
                    receiver_name_val_meta = n_val

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

        # Step 4: Spatial Fallback for fields spanning adjacent lines or fragmented OCR
        # 4a. Date & Time fallback
        if not date_val:
            d_val, t_val, d_ev, _ = DateParser.parse(norm_tokens)
            if d_val:
                date_val = d_val
                time_val = t_val or time_val
                date_conf = d_ev.confidence
                date_source = d_ev.source_text or ""

        # 4b. Transaction ID fallback
        if not ref_number:
            for t in norm_tokens:
                clean_num = re.sub(r"[^\d]", "", t.text)
                if len(clean_num) in [10, 11, 12] and (clean_num.startswith("202") or clean_num.startswith("201") or clean_num.startswith("200")):
                    ref_number = clean_num
                    ref_conf = t.confidence
                    ref_source = t.text
                    break

        # 4c. Sender / Receiver Account spatial fallback
        if not sender_account or not receiver_account:
            acc_candidates = cls._find_all_accounts_in_tokens(norm_tokens)
            if acc_candidates:
                if not sender_account and len(acc_candidates) >= 1:
                    sender_account = acc_candidates[0]
                if not receiver_account and len(acc_candidates) >= 2:
                    receiver_account = acc_candidates[1]

        # 4d. Recipient Name fallback across all tokens
        if not receiver_name:
            norm_n, raw_n, n_conf, n_val = ArabicNameProcessor.extract_name_from_tokens(norm_tokens, "receiver_name")
            if norm_n:
                receiver_name = norm_n
                receiver_name_raw = raw_n
                receiver_name_conf = n_conf
                receiver_name_val_meta = n_val

        # 4e. Amount fallback
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
                "raw_value": receiver_name_raw,
                "confidence": receiver_name_conf if receiver_name else 0.0,
                "source_text": receiver_name_raw or receiver_name,
                "evidence": f"Extracted from value region associated with label 'اسم المرسل اليه': '{receiver_name}'" if receiver_name else None,
                "label": "اسم المرسل اليه",
                "normalization_applied": bool(receiver_name_raw and receiver_name != receiver_name_raw),
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
            "amount": amount_conf if amount_val is not None else 0.0,
            "reference_number": ref_conf if ref_number else 0.0,
            "transaction_date": date_conf if date_val else 0.0,
            "transaction_time": 0.95 if time_val else 0.0,
            "sender_account": 0.95 if sender_account else 0.0,
            "receiver_account": 0.95 if receiver_account else 0.0,
            "receiver_name": receiver_name_conf if receiver_name else 0.0,
            "bank_name": 0.98,
        }

        # Weighted overall confidence calculation across ALL required fields
        field_weights = {
            "amount": 0.20,
            "reference_number": 0.20,
            "transaction_date": 0.15,
            "transaction_time": 0.05,
            "sender_account": 0.15,
            "receiver_account": 0.15,
            "receiver_name": 0.05,
            "bank_name": 0.05,
        }

        overall_conf = sum(field_confidences.get(k, 0.0) * w for k, w in field_weights.items())

        # Status determination: All major fields must be extracted for SUCCESS
        all_major_fields_present = (
            amount_val is not None
            and bool(ref_number)
            and bool(date_val)
            and bool(sender_account)
            and bool(receiver_account)
            and bool(receiver_name)
        )

        name_review_req = receiver_name is None or receiver_name_conf < 0.70

        if all_major_fields_present and overall_conf >= 0.85:
            status = ExtractionStatus.SUCCESS
            review_req = False
        elif amount_val is not None and ref_number:
            status = ExtractionStatus.PARTIAL_SUCCESS
            review_req = True
        else:
            status = ExtractionStatus.REVIEW_REQUIRED
            review_req = True

        warnings = []
        if is_future:
            warnings.append("Transaction date is flagged as a future date.")
            review_req = True
        if not amount_val:
            warnings.append("Amount could not be reliably extracted from the 'المبلغ' field.")
        if not date_val:
            warnings.append("Transaction date could not be extracted with confidence.")
        if not receiver_account:
            warnings.append("Receiver account number could not be extracted with confidence.")
        if not receiver_name:
            warnings.append("Receiver name could not be extracted with confidence.")

        raw_text = "\n".join(t.text for t in tokens)

        return StructuredReceiptData(
            amount=amount_val,
            formatted_amount=formatted_amount or (f"{amount_val:,.2f} {currency}" if amount_val is not None else None),
            currency=currency,
            sender_name=None,
            sender_name_raw=None,
            sender_name_confidence=None,
            sender_name_validation=None,
            sender_phone=sender_phone,
            sender_account=sender_account,
            receiver_name=receiver_name,
            receiver_name_raw=receiver_name_raw,
            receiver_name_confidence=receiver_name_conf if receiver_name else None,
            receiver_name_validation=receiver_name_val_meta if receiver_name else None,
            receiver_phone=receiver_phone,
            receiver_account=receiver_account,
            name_review_required=name_review_req,
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
    def _extract_bok_account(cls, line: List[OCRToken]) -> Optional[str]:
        """
        Extracts 16-digit Bank of Khartoum account formatted as 'XXXX XXXX XXXX XXXX'.
        Handles OCR missing leading/trailing zero (e.g. '373 1204 4436 0001' or '0913 0833 1734 001').
        """
        line_text = " ".join(t.text for t in line)
        digit_chunks = re.findall(r"\b\d{3,5}\b", line_text)

        if len(digit_chunks) == 4:
            c1, c2, c3, c4 = digit_chunks
            if len(c1) == 3:
                c1 = "0" + c1
            if len(c4) == 3:
                c4 = "0" + c4
            if len(c1) == 4 and len(c2) == 4 and len(c3) == 4 and len(c4) == 4:
                return f"{c1} {c2} {c3} {c4}"

        continuous = re.search(r"\b\d{16}\b", line_text.replace(" ", ""))
        if continuous:
            raw = continuous.group(0)
            return f"{raw[:4]} {raw[4:8]} {raw[8:12]} {raw[12:]}"

        return None

    @classmethod
    def _find_all_accounts_in_tokens(cls, tokens: List[OCRToken]) -> List[str]:
        """Scans all tokens to identify all 16-digit account candidates."""
        accounts = []
        digit_tokens = [t.text.strip() for t in tokens if re.match(r"^\d{3,5}$", t.text.strip())]
        for i in range(len(digit_tokens) - 3):
            sub = digit_tokens[i : i + 4]
            c1, c2, c3, c4 = sub
            if len(c1) == 3:
                c1 = "0" + c1
            if len(c4) == 3:
                c4 = "0" + c4
            if len(c1) == 4 and len(c2) == 4 and len(c3) == 4 and len(c4) == 4:
                acc = f"{c1} {c2} {c3} {c4}"
                if acc not in accounts:
                    accounts.append(acc)
        return accounts

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
