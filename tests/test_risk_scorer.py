"""
Tests for Layer 2: Risk Scoring
"""
import pytest
from datetime import datetime, timedelta


class TestRiskScorer:
    """Test suite for risk scoring signals."""

    def test_round_number_score_high_for_round(self):
        """Round numbers like $1000 should get high score."""
        from api.services.risk_scorer import RiskScorer

        scorer = RiskScorer()

        score, reason = scorer._score_round_number(1000)

        assert score > 50  # Should be flagged

    def test_round_number_score_low_for_irregular(self):
        """Non-round numbers should get low score."""
        from api.services.risk_scorer import RiskScorer

        scorer = RiskScorer()

        score, reason = scorer._score_round_number(1234.56)

        assert score < 50  # Should not be flagged

    def test_unusual_time_late_night(self):
        """Transactions at 3 AM should be flagged as unusual time."""
        from api.services.risk_scorer import RiskScorer

        scorer = RiskScorer()

        late_night = datetime(2026, 2, 20, 3, 0, 0)  # 3 AM Friday
        score, reason = scorer._score_unusual_time(late_night)

        assert score > 0  # Should have some score for late night

    def test_unusual_time_business_hours(self):
        """Transactions during business hours should have low unusual_time score."""
        from api.services.risk_scorer import RiskScorer

        scorer = RiskScorer()

        # 2 PM on a weekday
        normal_time = datetime(2026, 2, 20, 14, 0, 0)  # Friday 2 PM
        score, reason = scorer._score_unusual_time(normal_time)

        # Business hours should have lower score than late night
        late_night = datetime(2026, 2, 20, 3, 0, 0)
        late_score, _ = scorer._score_unusual_time(late_night)

        assert score <= late_score

    def test_unusual_time_weekend(self):
        """Weekend transactions should be flagged."""
        from api.services.risk_scorer import RiskScorer

        scorer = RiskScorer()

        saturday = datetime(2026, 2, 21, 14, 0, 0)  # Saturday
        score, reason = scorer._score_unusual_time(saturday)

        assert score > 0  # Weekend should have some score

    def test_weights_sum_to_one(self):
        """Risk signal weights should sum to 1.0 for proper normalization."""
        from api.services.risk_scorer import RiskScorer

        scorer = RiskScorer()

        total_weight = sum(scorer.WEIGHTS.values())

        assert abs(total_weight - 1.0) < 0.01, f"Weights sum to {total_weight}, expected 1.0"

    def test_all_signals_present(self):
        """All expected risk signals should be defined."""
        from api.services.risk_scorer import RiskScorer

        scorer = RiskScorer()

        expected_signals = [
            'amount_deviation',
            'new_vendor',
            'unusual_time',
            'round_number',
            'velocity',
            'category_mismatch',
            'ml_score',
            'peer_comparison'
        ]

        for signal in expected_signals:
            assert signal in scorer.WEIGHTS, f"Missing signal: {signal}"

    def test_new_vendor_high_amount_flagged(self):
        """New vendor with high amount should be flagged."""
        from api.services.risk_scorer import RiskScorer

        scorer = RiskScorer()

        score, reason = scorer._score_new_vendor(
            merchant='Brand New Store',
            amount=5000,
            known_vendors=[]  # No known vendors
        )

        assert score > 50  # New vendor should be flagged

    def test_known_vendor_not_flagged(self):
        """Known vendor should have lower score."""
        from api.services.risk_scorer import RiskScorer

        scorer = RiskScorer()

        score, reason = scorer._score_new_vendor(
            merchant='Regular Store',
            amount=100,
            known_vendors=['Regular Store', 'Another Store']
        )

        assert score < 50  # Known vendor should not be flagged

    def test_high_velocity_flagged(self):
        """Multiple transactions in short time should increase score."""
        from api.services.risk_scorer import RiskScorer

        scorer = RiskScorer()

        # Simulate 5 transactions in last hour
        recent_txns = [
            {'amount': 100, 'created_at': datetime.utcnow() - timedelta(minutes=i*10)}
            for i in range(5)
        ]

        score, reason = scorer._score_velocity(recent_txns)

        assert score > 0  # Should have some velocity score

    def test_low_velocity_not_flagged(self):
        """Single transaction should not trigger velocity flag."""
        from api.services.risk_scorer import RiskScorer

        scorer = RiskScorer()

        recent_txns = []  # No recent transactions

        score, reason = scorer._score_velocity(recent_txns)

        assert score == 0  # No velocity issue

    def test_amount_deviation_high(self):
        """Amount far from mean should be flagged."""
        from api.services.risk_scorer import RiskScorer

        scorer = RiskScorer()

        # User avg is $100, std is $20, transaction is $500
        score, reason = scorer._score_amount_deviation(
            amount=500,
            avg=100,
            std=20
        )

        assert score > 50  # High deviation should be flagged

    def test_amount_deviation_normal(self):
        """Amount close to mean should have low score."""
        from api.services.risk_scorer import RiskScorer

        scorer = RiskScorer()

        # User avg is $100, std is $20, transaction is $110
        score, reason = scorer._score_amount_deviation(
            amount=110,
            avg=100,
            std=20
        )

        assert score < 50  # Normal deviation should not be flagged

    def test_severity_critical(self):
        """Score >= 90 should be critical severity."""
        from api.services.risk_scorer import RiskScorer

        scorer = RiskScorer()

        severity = scorer.get_severity(95)

        assert severity == 'critical'

    def test_severity_high(self):
        """Score 70-89 should be high severity."""
        from api.services.risk_scorer import RiskScorer

        scorer = RiskScorer()

        severity = scorer.get_severity(75)

        assert severity == 'high'

    def test_severity_medium(self):
        """Score 40-69 should be medium severity."""
        from api.services.risk_scorer import RiskScorer

        scorer = RiskScorer()

        severity = scorer.get_severity(50)

        assert severity == 'medium'

    def test_severity_low(self):
        """Score < 40 should be low severity."""
        from api.services.risk_scorer import RiskScorer

        scorer = RiskScorer()

        severity = scorer.get_severity(20)

        assert severity == 'low'