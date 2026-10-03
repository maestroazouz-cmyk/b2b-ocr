import re
from typing import List, Optional, Tuple
from app.models.schemas import OCRToken, EvidenceField


class BankParser:
    """
    Identifies Sudanese issuing banks and payment wallets from voucher text.
    Evidence-based rule:
    - Never automatically map a wallet name (e.g. 'فوري / Fawry') to a bank unless the bank is explicitly present.
    - Captures bank_name and wallet_name as separate distinct entities.
    """

    # Explicit Sudanese Banks
    SUDANESE_BANKS = {
        "Bank of Khartoum": ["بنك الخرطوم", "بنكك", "Bankak", "Bank of Khartoum", "BOK"],
        "Faisal Islamic Bank": ["بنك فيصل الإسلامي", "بنك فيصل الاسلامي", "Faisal Islamic Bank", "FIB"],
        "Omdurman National Bank": ["بنك أمدرمان الوطني", "بنك امدرمان الوطني", "Omdurman National Bank", "ONB"],
        "Nile Bank": ["بنك النيل", "بنك النيل للتجارة", "Nile Bank"],
        "Al Baraka Bank": ["بنك البركة", "Al Baraka Bank", "Albaraka"],
        "Al Salam Bank": ["بنك السلام", "Al Salam Bank", "Alsalam"],
        "Sudanese Islamic Bank": ["البنك الإسلامي السوداني", "البنك الاسلامي السوداني", "Sudanese Islamic Bank"],
        "Blue Nile Mashreq Bank": ["بنك النيل الأزرق المشرق", "بنك النيل الازرق المشرق", "Blue Nile Mashreq"],
        "United Capital Bank": ["البنك الأهلي المتحد", "البنك الاهلي المتحد", "United Capital Bank"],
        "Workers National Bank": ["بنك العمال الوطني", "Workers National Bank"],
        "Farmer's Commercial Bank": ["البنك التجاري للإنتاج والمزارعين", "بنك المزارع التجاري"],
        "Saudi Sudanese Bank": ["البنك السعودي السوداني", "Saudi Sudanese Bank"],
        "Tadamon Islamic Bank": ["بنك التضامن الإسلامي", "بنك التضامن الاسلامي", "Tadamon Islamic Bank"],
    }

    # Distinct Wallets / Channels
    WALLETS = {
        "Bankak": ["بنكك", "Bankak"],
        "Fawry": ["فوري", "Fawry"],
        "O-Cash": ["أوكاش", "اوكاش", "O-Cash", "OCash"],
        "Bashaer": ["بشائر", "Bashaer"],
        "Sayer": ["ساير", "Sayer"],
        "Cashi": ["كاشي", "Cashi"],
    }

    @classmethod
    def parse(cls, tokens: List[OCRToken]) -> Tuple[Optional[str], Optional[str], EvidenceField]:
        """
        Returns (bank_name, wallet_name, evidence_field)
        """
        full_text = " ".join(t.text for t in tokens)

        detected_bank: Optional[str] = None
        detected_wallet: Optional[str] = None
        best_conf = 0.0
        source_tok = ""

        # 1. Detect Explicit Bank
        for bank_canonical, aliases in cls.SUDANESE_BANKS.items():
            for alias in aliases:
                for token in tokens:
                    if alias in token.text:
                        detected_bank = bank_canonical
                        best_conf = max(best_conf, min(0.98, token.confidence))
                        source_tok = token.text
                        break
                if detected_bank:
                    break
            if detected_bank:
                break

        # 2. Detect Wallet / Service Name independently
        for wallet_canonical, aliases in cls.WALLETS.items():
            for alias in aliases:
                for token in tokens:
                    if alias in token.text:
                        detected_wallet = wallet_canonical
                        break
                if detected_wallet:
                    break
            if detected_wallet:
                break

        evidence = EvidenceField(
            value=detected_bank,
            confidence=best_conf,
            source_text=source_tok if detected_bank else None,
            evidence=f"Identified Bank: {detected_bank} (Wallet: {detected_wallet})" if detected_bank else None,
        )

        return detected_bank, detected_wallet, evidence
