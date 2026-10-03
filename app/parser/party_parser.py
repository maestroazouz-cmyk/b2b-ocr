import re
from typing import List, Optional, Tuple
from app.models.schemas import OCRToken, EvidenceField


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
        "إلى",
        "المستقبل",
        "Beneficiary",
        "Receiver",
        "To",
        "Payee",
    ]

    PLACEHOLDER_NAMES = [
        "unknown",
        "n/a",
        "not available",
        "customer",
        "sender",
        "receiver",
        "client",
        "user",
        "test user",
        "sample",
        "null",
        "none",
        "غير معروف",
        "العميل",
        "المستفيد",
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

        # 1. Extract Sender
        for i, token in enumerate(tokens):
            norm_text = token.text.strip()
            lbl = next((l for l in cls.SENDER_LABELS if norm_text.startswith(l) or f" {l} " in f" {norm_text} "), None)

            if lbl:
                cand = norm_text.replace(lbl, "").strip(" :-\t")
                if cls._is_valid_name(cand):
                    sender_name = cand
                    sender_ev = EvidenceField(
                        value=cand,
                        confidence=min(0.95, token.confidence),
                        source_text=token.text,
                        evidence=f"Sender label '{lbl}' in token: '{token.text}'",
                    )
                    break

                for j in range(i + 1, min(i + 3, len(tokens))):
                    adj = tokens[j]
                    adj_text = adj.text.strip(" :-\t")
                    if cls._is_valid_name(adj_text):
                        sender_name = adj_text
                        conf = min(0.92, (token.confidence + adj.confidence) / 2.0)
                        sender_ev = EvidenceField(
                            value=adj_text,
                            confidence=conf,
                            source_text=f"{token.text} -> {adj.text}",
                            evidence=f"Sender label '{lbl}' followed by '{adj.text}'",
                        )
                        break
                if sender_name:
                    break

        # 2. Extract Receiver
        for i, token in enumerate(tokens):
            norm_text = token.text.strip()
            lbl = next((l for l in cls.RECEIVER_LABELS if norm_text.startswith(l) or f" {l} " in f" {norm_text} "), None)

            if lbl:
                cand = norm_text.replace(lbl, "").strip(" :-\t")
                if cls._is_valid_name(cand):
                    receiver_name = cand
                    receiver_ev = EvidenceField(
                        value=cand,
                        confidence=min(0.95, token.confidence),
                        source_text=token.text,
                        evidence=f"Receiver label '{lbl}' in token: '{token.text}'",
                    )
                    break

                for j in range(i + 1, min(i + 3, len(tokens))):
                    adj = tokens[j]
                    adj_text = adj.text.strip(" :-\t")
                    if cls._is_valid_name(adj_text):
                        receiver_name = adj_text
                        conf = min(0.92, (token.confidence + adj.confidence) / 2.0)
                        receiver_ev = EvidenceField(
                            value=adj_text,
                            confidence=conf,
                            source_text=f"{token.text} -> {adj.text}",
                            evidence=f"Receiver label '{lbl}' followed by '{adj.text}'",
                        )
                        break
                if receiver_name:
                    break

        # 3. Account numbers
        for token in tokens:
            if "حساب" in token.text or "Account" in token.text:
                acc_match = re.search(r"\b\d{8,20}\b", token.text)
                if acc_match:
                    if not sender_acc:
                        sender_acc = acc_match.group(0)
                    elif not receiver_acc:
                        receiver_acc = acc_match.group(0)

        return sender_name, sender_ev, receiver_name, receiver_ev, sender_acc, receiver_acc

    @classmethod
    def _is_valid_name(cls, text: str) -> bool:
        if not text or len(text) < 3:
            return False
        if text.lower() in cls.PLACEHOLDER_NAMES:
            return False
        # Must contain at least some Arabic or Latin alphabetic characters
        if not re.search(r"[a-zA-Z\u0600-\u06FF]", text):
            return False
        # Should not be pure digits or date string
        if re.match(r"^[\d\-\/\.\s,]+$", text):
            return False
        return True
