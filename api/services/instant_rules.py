"""
Instant Rules - Hard blocks evaluated in real-time (<100ms)
These rules can BLOCK transactions immediately
"""

from datetime import datetime, time
from typing import Dict, List, Tuple


class InstantRules:
    """
    Hard rules that can block transactions immediately.
    Matches Ramp's real-time authorization approach.
    """

    def __init__(self):
        # Configurable thresholds
        self.blocked_vendors = [
            "CASINO", "GAMBLING", "LOTTERY", "ADULT", "XXX"
        ]
        self.blocked_mccs = [
            "7995",  # Gambling
            "5933",  # Pawn shops
            "5944",  # Jewelry (high fraud risk)
        ]
        self.max_single_transaction = 10000  # $10k hard limit
        self.business_hours = (time(6, 0), time(22, 0))  # 6am - 10pm

    def evaluate(self, transaction: Dict) -> Tuple[bool, List[str]]:
        """
        Evaluate transaction against hard rules.

        Returns:
            (should_block, list of reasons)
        """
        violations = []

        # Rule 1: Blocked vendor check
        merchant = (transaction.get("merchant_name") or "").upper()
        for blocked in self.blocked_vendors:
            if blocked in merchant:
                violations.append(f"BLOCKED_VENDOR: Merchant contains '{blocked}'")

        # Rule 2: Blocked MCC (Merchant Category Code)
        mcc = transaction.get("merchant_category_code")
        if mcc in self.blocked_mccs:
            violations.append(f"BLOCKED_CATEGORY: MCC {mcc} is restricted")

        # Rule 3: Hard amount limit
        amount = transaction.get("amount", 0)
        if amount > self.max_single_transaction:
            violations.append(f"OVER_LIMIT: ${amount} exceeds ${self.max_single_transaction} limit")

        # Rule 4: Duplicate transaction (same merchant + amount within 5 min)
        # Note: This requires DB lookup, handled separately

        should_block = len(violations) > 0
        return should_block, violations

    def check_duplicate(self, transaction: Dict, recent_transactions: List[Dict]) -> Tuple[bool, str]:
        """
        Check for duplicate transactions.

        Args:
            transaction: Current transaction
            recent_transactions: Transactions from same user in last 10 minutes

        Returns:
            (is_duplicate, reason)
        """
        current_amount = transaction.get("amount")
        current_merchant = (transaction.get("merchant_name") or "").lower()

        for recent in recent_transactions:
            if (recent.get("amount") == current_amount and
                    (recent.get("merchant_name") or "").lower() == current_merchant):
                return True, f"DUPLICATE: Same amount (${current_amount}) at {current_merchant} within 10 minutes"

        return False, ""

    def check_receipt_mismatch(self, transaction_amount: float, receipt_amount: float, tolerance: float = 0.01) -> \
    Tuple[bool, str]:
        """
        Check if receipt amount matches transaction amount.
        This is Ramp's highlighted feature.

        Args:
            transaction_amount: Amount from card transaction
            receipt_amount: Amount from submitted receipt
            tolerance: Allowed difference (default 1%)

        Returns:
            (is_mismatch, reason)
        """
        if receipt_amount is None:
            return False, ""  # No receipt submitted yet

        diff = abs(transaction_amount - receipt_amount)
        diff_percent = diff / transaction_amount if transaction_amount > 0 else 0

        if diff_percent > tolerance:
            return True, f"RECEIPT_MISMATCH: Transaction ${transaction_amount} vs Receipt ${receipt_amount} (diff: ${diff:.2f})"

        return False, ""


# Singleton instance
instant_rules = InstantRules()