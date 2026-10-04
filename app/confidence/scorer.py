from typing import Dict


class ConfidenceScorer:
    """
    Weighted confidence scoring based on visual OCR token accuracy,
    field proximity, and presence/absence of all required structured fields.
    """

    WEIGHTS = {
        "amount": 0.20,
        "reference_number": 0.20,
        "transaction_date": 0.15,
        "transaction_time": 0.05,
        "sender_account": 0.15,
        "receiver_account": 0.15,
        "receiver_name": 0.05,
        "bank_name": 0.05,
    }

    @classmethod
    def calculate_overall_confidence(
        cls, field_confidences: Dict[str, float], image_quality_score: float
    ) -> float:
        total_score = 0.0
        total_weight = 0.0

        for field, weight in cls.WEIGHTS.items():
            conf = field_confidences.get(field, 0.0)
            total_score += conf * weight
            total_weight += weight

        base_conf = total_score / total_weight if total_weight > 0 else 0.0

        # Adjust slightly if image quality is degraded
        if image_quality_score < 0.60:
            base_conf *= max(0.5, image_quality_score)

        return min(1.0, max(0.0, base_conf))
