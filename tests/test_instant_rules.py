"""
Tests for Layer 1: Instant Rules (Hard Blocks)
"""
import pytest
from datetime import datetime, timedelta


class TestInstantRules:
    """Test suite for instant rule checks."""

    def test_amount_over_limit_blocked(self):
        """Transactions over $10,000 should be blocked."""
        from api.services.instant_rules import InstantRules

        rules = InstantRules()
        transaction = {
            'amount': 15000,
            'user_id': 'user_001',
            'merchant_name': 'Big Purchase Inc',
            'merchant_category_code': '5999'
        }

        blocked, reasons = rules.evaluate(transaction)

        assert blocked == True
        assert any('limit' in r.lower() or 'amount' in r.lower() for r in reasons)

    def test_amount_under_limit_passes(self):
        """Transactions under $10,000 should pass instant rules."""
        from api.services.instant_rules import InstantRules

        rules = InstantRules()
        transaction = {
            'amount': 5000,
            'user_id': 'user_001',
            'merchant_name': 'Normal Store',
            'merchant_category_code': '5999'
        }

        blocked, reasons = rules.evaluate(transaction)

        assert blocked == False

    def test_gambling_mcc_blocked(self):
        """Gambling merchant category (7995) should be blocked."""
        from api.services.instant_rules import InstantRules

        rules = InstantRules()
        transaction = {
            'amount': 100,
            'user_id': 'user_001',
            'merchant_name': 'Lucky Casino',
            'merchant_category_code': '7995'
        }

        blocked, reasons = rules.evaluate(transaction)

        assert blocked == True

    def test_pawn_shop_mcc_blocked(self):
        """Pawn shop merchant category (5933) should be blocked."""
        from api.services.instant_rules import InstantRules

        rules = InstantRules()
        transaction = {
            'amount': 50,
            'user_id': 'user_001',
            'merchant_name': 'Quick Cash Pawn',
            'merchant_category_code': '5933'
        }

        blocked, reasons = rules.evaluate(transaction)

        assert blocked == True

    def test_normal_mcc_passes(self):
        """Normal merchant categories should pass."""
        from api.services.instant_rules import InstantRules

        rules = InstantRules()
        transaction = {
            'amount': 100,
            'user_id': 'user_001',
            'merchant_name': 'Office Supplies',
            'merchant_category_code': '5943'
        }

        blocked, reasons = rules.evaluate(transaction)

        assert blocked == False

    def test_edge_case_exactly_at_limit(self):
        """Transaction exactly at $10,000 should pass."""
        from api.services.instant_rules import InstantRules

        rules = InstantRules()
        transaction = {
            'amount': 10000,
            'user_id': 'user_001',
            'merchant_name': 'Expensive Store',
            'merchant_category_code': '5999'
        }

        blocked, reasons = rules.evaluate(transaction)

        # Exactly at limit should pass (only OVER limit is blocked)
        assert blocked == False

    def test_duplicate_detection(self):
        """Duplicate transactions should be detected."""
        from api.services.instant_rules import InstantRules

        rules = InstantRules()

        transaction = {
            'amount': 100,
            'user_id': 'user_001',
            'merchant_name': 'Coffee Shop'
        }

        recent_transactions = [
            {
                'amount': 100,
                'user_id': 'user_001',
                'merchant_name': 'Coffee Shop',
                'created_at': datetime.utcnow() - timedelta(minutes=5)
            }
        ]

        is_duplicate, reason = rules.check_duplicate(transaction, recent_transactions)

        assert is_duplicate == True

    def test_non_duplicate_passes(self):
        """Different transactions should not be flagged as duplicates."""
        from api.services.instant_rules import InstantRules

        rules = InstantRules()

        transaction = {
            'amount': 200,
            'user_id': 'user_001',
            'merchant_name': 'Restaurant'
        }

        recent_transactions = [
            {
                'amount': 100,
                'user_id': 'user_001',
                'merchant_name': 'Coffee Shop',
                'created_at': datetime.utcnow() - timedelta(minutes=5)
            }
        ]

        is_duplicate, reason = rules.check_duplicate(transaction, recent_transactions)

        assert is_duplicate == False

    def test_receipt_mismatch_detected(self):
        """Receipt amount mismatch should be detected."""
        from api.services.instant_rules import InstantRules

        rules = InstantRules()

        is_mismatch, reason = rules.check_receipt_mismatch(
            transaction_amount=100.00,
            receipt_amount=150.00
        )

        assert is_mismatch == True

    def test_receipt_match_passes(self):
        """Matching receipt amount should pass."""
        from api.services.instant_rules import InstantRules

        rules = InstantRules()

        is_mismatch, reason = rules.check_receipt_mismatch(
            transaction_amount=100.00,
            receipt_amount=100.50  # Within tolerance
        )

        assert is_mismatch == False