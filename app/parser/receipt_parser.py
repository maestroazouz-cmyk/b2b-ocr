import re
from typing import List, Dict, Any, Optional
try:
    import numpy as np
except ImportError:
    np = None

from app.models.schemas import (
    OCRToken,
    StructuredReceiptData,
    ImageQuality,
    ExtractionStatus,
)
from app.parser.amount_parser import AmountParser
from app.parser.phone_parser import PhoneParser
from app.parser.transaction_parser import TransactionParser
from app.parser.date_parser import DateParser
from app.parser.party_parser import PartyParser
from app.parser.bank_parser import BankParser
from app.parser.bok_layout_parser import BankOfKhartoumLayoutParser
from app.confidence.scorer import ConfidenceScorer


class SudaneseReceiptParser:
    """
    Orchestrates two-stage receipt OCR parsing:
    1. Deterministic Structured Layout Detection (e.g. Bank of Khartoum vouchers).
    2. Robust Generic Sub-Parser Pipeline for multi-bank and mobile wallet vouchers.
    Enforces strict anti-hallucination guardrails and generates full evidence map.
    """

    @classmethod
    def parse(
        cls,
        tokens: List[OCRToken],
        image_quality: ImageQuality,
        image_bgr: Optional[Any] = None,
        ocr_engine: Optional[Any] = None,
    ) -> StructuredReceiptData:
        # Check if receipt matches structured Bank of Khartoum layout
        if BankOfKhartoumLayoutParser.matches_layout(tokens):
            return BankOfKhartoumLayoutParser.parse_bok_voucher(
                tokens, image_quality, image_bgr, ocr_engine
            )

        # Fallback to Generic Multi-bank / Wallet Parser
        raw_text = "\n".join(t.text for t in tokens)

        # 1. Sub-parsers
        amount_val, formatted_amount, currency, amount_ev = AmountParser.parse(tokens)
        sender_phone, phone_ev = PhoneParser.parse(tokens)
        ref_num, ref_ev = TransactionParser.parse(tokens)
        tx_date, tx_time, date_ev, val_meta = DateParser.parse(tokens)
        (
            sender_name,
            sender_ev,
            receiver_name,
            receiver_ev,
            sender_acc,
            receiver_acc,
        ) = PartyParser.parse(tokens)
        bank_name, wallet_name, bank_ev = BankParser.parse(tokens)

        # 2. Extract Narration / Remarks
        narration = cls._extract_narration(tokens)

        # 3. Evidence Map
        evidence_map: Dict[str, Any] = {
            "amount": amount_ev.model_dump(),
            "reference_number": ref_ev.model_dump(),
            "transaction_date": date_ev.model_dump(),
            "sender_name": sender_ev.model_dump(),
            "receiver_name": receiver_ev.model_dump(),
            "bank_name": bank_ev.model_dump(),
            "sender_phone": phone_ev.model_dump(),
        }

        # 4. Field Confidences
        field_confidences = {
            "amount": amount_ev.confidence,
            "reference_number": ref_ev.confidence,
            "transaction_date": date_ev.confidence,
            "sender_name": sender_ev.confidence,
            "receiver_name": receiver_ev.confidence,
            "bank_name": bank_ev.confidence,
            "sender_phone": phone_ev.confidence,
        }

        # 5. Overall Confidence Calculation
        overall_conf = ConfidenceScorer.calculate_overall_confidence(
            field_confidences, image_quality.image_quality_score
        )

        # 6. Status Determination
        has_phone = bool(sender_phone)
        has_ref = bool(ref_num)
        has_amount = amount_val is not None

        if has_phone and has_ref and has_amount and overall_conf >= 0.80:
            status = ExtractionStatus.SUCCESS
        elif (has_amount and has_phone) or (has_amount and has_ref):
            status = ExtractionStatus.PARTIAL_SUCCESS
        elif has_amount or has_ref or has_phone:
            status = ExtractionStatus.REVIEW_REQUIRED
        else:
            status = ExtractionStatus.MANUAL_ENTRY

        warnings = []
        if val_meta.is_future_date:
            warnings.append(val_meta.date_interpretation_notes)
        if not ref_num:
            warnings.append("Transaction reference number is not clearly visible on the receipt. Manual verification required.")
        if not tx_date:
            warnings.append("Transaction execution date could not be identified with confidence. Human verification required.")
        if not amount_val:
            warnings.append("Amount could not be identified with confidence. Human verification required.")
        if image_quality.quality_status.value in ["POOR", "UNUSABLE"]:
            warnings.append(f"Image quality is {image_quality.quality_status.value.lower()} ({', '.join(image_quality.quality_issues) or 'defects detected'}).")

        review_required = (
            status != ExtractionStatus.SUCCESS
            or val_meta.is_future_date
            or not ref_num
            or not tx_date
            or not amount_val
            or overall_conf < 0.85
        )

        return StructuredReceiptData(
            amount=amount_val,
            formatted_amount=formatted_amount,
            currency=currency,
            sender_name=sender_name,
            sender_phone=sender_phone,
            sender_account=sender_acc,
            receiver_name=receiver_name,
            receiver_phone=None,
            receiver_account=receiver_acc,
            transaction_date=tx_date,
            transaction_time=tx_time,
            reference_number=ref_num,
            transaction_id=ref_num,
            bank_name=bank_name,
            wallet_name=wallet_name,
            narration=narration,
            transaction_type="BANK_TRANSFER" if bank_name else "MOBILE_WALLET" if wallet_name else "TRANSFER",
            transaction_status="SUCCESS" if has_amount and (has_ref or has_phone) else "PENDING",
            merchant_name="Zain Telecom B2B",
            raw_text=raw_text,
            image_quality=image_quality,
            field_confidences=field_confidences,
            overall_confidence=round(overall_conf, 2),
            warnings=warnings,
            review_required=review_required,
            extraction_version="b2b-ocr-v2",
            evidence_map=evidence_map,
            validation_metadata=val_meta,
            result_status=status,
        )

    @classmethod
    def _extract_narration(cls, tokens: List[OCRToken]) -> str:
        narration_kws = ["البيان", "الغرض", "ملاحظات", "التعليق", "Remarks", "Narration", "Purpose"]
        for i, token in enumerate(tokens):
            for kw in narration_kws:
                if kw in token.text:
                    clean = token.text.replace(kw, "").strip(" :-\t")
                    if len(clean) > 3:
                        return clean
                    if i + 1 < len(tokens):
                        return tokens[i + 1].text.strip()
        return ""
