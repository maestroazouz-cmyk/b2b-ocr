import re
from typing import List, Optional, Tuple
from app.models.schemas import OCRToken, EvidenceField
from app.parser.arabic_name_processor import ArabicNameProcessor


class PartyParser:
    """
    Extracts Sender (المحول / من) and Receiver / Beneficiary (المستفيد / إلى) names and accounts.
    Strict anti-hallucination rules:
    - Rejects placeholder names ('unknown', 'customer', 'n/a', 'user').
    - Only extracts when explicitly labeled on the receipt.
    - If sender/receiver is absent, returns null with confidence = 0.
    """

    SENDER_LABELS = [
        "اسم المحول",
        "المحول",
        "اسم الراسل",
        "الراسل",
        "من",
        "اسم العميل",
        "المرسل",
        "Sender",
        "From",
        "Payer",
    ]

    RECEIVER_LABELS = [
        "اسم المستفيد",
        "المستفيد",
        "المحول إليه",
        "المحول اليه",
        "إلى",
        "الى",
        "المستقبل",
        "Beneficiary",
        "Receiver",
        "To",
        "Payee",
    ]

    @classmethod
    def parse(
        cls, tokens: List[OCRToken]
    ) -> Tuple[
        Optional[str],
        EvidenceField,
        Optional[str],
        EvidenceField,
        Optional[str],
        Optional[str],
    ]:
        """
        Returns (sender_name, sender_ev, receiver_name, receiver_ev, sender_acc, receiver_acc)
        """
        sender_name: Optional[str] = None
        receiver_name: Optional[str] = None
        sender_acc: Optional[str] = None
        receiver_acc: Optional[str] = None

        sender_ev = EvidenceField(value=None, confidence=0.0)
        receiver_ev = EvidenceField(value=None, confidence=0.0)

        # 1. Extract Sender via ArabicNameProcessor
        s_norm, s_raw, s_conf, s_val = ArabicNameProcessor.extract_name_from_tokens(tokens, "sender_name")
        if s_norm:
            sender_name = s_norm
            sender_ev = EvidenceField(
                value=s_norm,
                raw_value=s_raw,
                confidence=s_conf,
                source_text=s_raw or s_norm,
                evidence=f"Extracted sender name: '{s_norm}'",
                label="اسم المرسل",
                normalization_applied=bool(s_raw and s_norm != s_raw),
            )

        # 2. Extract Receiver via ArabicNameProcessor
        r_norm, r_raw, r_conf, r_val = ArabicNameProcessor.extract_name_from_tokens(tokens, "receiver_name")
        if r_norm:
            receiver_name = r_norm
            receiver_ev = EvidenceField(
                value=r_norm,
                raw_value=r_raw,
                confidence=r_conf,
                source_text=r_raw or r_norm,
                evidence=f"Extracted receiver name: '{r_norm}'",
                label="اسم المستفيد",
                normalization_applied=bool(r_raw and r_norm != r_raw),
            )

        # 3. Account numbers
        for token in tokens:
            if "حساب" in token.text or "Account" in token.text:
                acc_match = re.search(r"\b\d{8,20}\b", token.text.replace(" ", ""))
                if acc_match:
                    raw_acc = acc_match.group(0)
                    if len(raw_acc) == 16:
                        formatted = f"{raw_acc[:4]} {raw_acc[4:8]} {raw_acc[8:12]} {raw_acc[12:]}"
                    else:
                        formatted = raw_acc

                    if not sender_acc:
                        sender_acc = formatted
                    elif not receiver_acc and formatted != sender_acc:
                        receiver_acc = formatted

        return sender_name, sender_ev, receiver_name, receiver_ev, sender_acc, receiver_acc
