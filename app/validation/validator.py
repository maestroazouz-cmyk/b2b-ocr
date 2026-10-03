from typing import List, Tuple
from app.models.schemas import StructuredReceiptData


class ReceiptValidator:
    """
    Business validation rules for Sudanese corporate collections:
    - Amount > 0 and reasonable upper threshold.
    - Valid 10-digit Sudanese phone (09... / 01...).
    - Temporal validity (Africa/Khartoum).
    - Checks required financial fields without mutating source values.
    """

    @classmethod
    def validate(cls, data: StructuredReceiptData) -> Tuple[bool, List[str]]:
        warnings: List[str] = []
        is_valid = True

        # 1. Amount Check
        if data.amount is None:
            warnings.append("Amount is missing.")
            is_valid = False
        elif data.amount <= 0:
            warnings.append("Invalid payment amount (<= 0).")
            is_valid = False
        elif data.amount > 500_000_000:
            warnings.append("Payment amount exceeds standard daily limit (500M SDG).")

        # 2. Phone Check
        if data.sender_phone:
            if not (data.sender_phone.startswith("09") or data.sender_phone.startswith("01")) or len(data.sender_phone) != 10:
                warnings.append(f"Phone number '{data.sender_phone}' does not match standard 10-digit Sudanese format.")

        # 3. Reference Number Check
        if not data.reference_number:
            warnings.append("Transaction reference number is missing.")
            is_valid = False

        # 4. Date Check
        if data.validation_metadata and data.validation_metadata.is_future_date:
            warnings.append(f"Future transaction date flagged: {data.transaction_date}")

        return is_valid, warnings
