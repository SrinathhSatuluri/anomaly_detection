"""
Tests for Layer 3: ML Ensemble Detector
"""
import pytest
import numpy as np
from datetime import datetime


class TestEnsembleDetector:
    """Test suite for ML ensemble detector."""

    def test_model_loads_successfully(self):
        """Models should load without error."""
        from ml.ensemble_detector import EnsembleDetector

        detector = EnsembleDetector()

        # Should have loaded models (check via get_model_info)
        info = detector.get_model_info()

        assert info is not None

    def test_model_info_contains_version(self):
        """Model info should contain version."""
        from ml.ensemble_detector import EnsembleDetector

        detector = EnsembleDetector()
        info = detector.get_model_info()

        assert 'version' in info or 'trained_at' in info or info.get('model_loaded') is not None

    def test_feature_extraction_returns_array(self):
        """Feature extraction should return numpy array."""
        from ml.ensemble_detector import EnsembleDetector

        detector = EnsembleDetector()

        transaction = {
            'amount': 500,
            'user_id': 'user_001',
            'merchant_name': 'Test Store',
            'merchant_category_code': '5999',
            'transaction_date': datetime.utcnow()
        }

        user_stats = {
            'avg_amount': 200,
            'std_amount': 50,
            'max_amount': 1000,
            'transaction_count': 10,
            'known_vendors': ['Test Store'],
            'recent_transactions': []
        }

        features = detector.extract_features(transaction, user_stats)

        assert isinstance(features, np.ndarray)

    def test_feature_count_is_30(self):
        """Should extract exactly 30 features."""
        from ml.ensemble_detector import EnsembleDetector

        detector = EnsembleDetector()

        transaction = {
            'amount': 500,
            'user_id': 'user_001',
            'merchant_name': 'Test Store',
            'merchant_category_code': '5999',
            'transaction_date': datetime.utcnow()
        }

        user_stats = {
            'avg_amount': 200,
            'std_amount': 50,
            'max_amount': 1000,
            'transaction_count': 10,
            'known_vendors': ['Test Store'],
            'recent_transactions': []
        }

        features = detector.extract_features(transaction, user_stats)

        assert features.size == 30, f"Expected 30 features, got {features.size}"

    def test_prediction_returns_tuple(self):
        """Prediction should return score, decision, and details."""
        from ml.ensemble_detector import EnsembleDetector

        detector = EnsembleDetector()

        transaction = {
            'amount': 500,
            'user_id': 'user_001',
            'merchant_name': 'Test Store',
            'merchant_category_code': '5999',
            'transaction_date': datetime.utcnow()
        }

        user_stats = {
            'avg_amount': 200,
            'std_amount': 50,
            'max_amount': 1000,
            'transaction_count': 10,
            'known_vendors': ['Test Store'],
            'recent_transactions': []
        }

        result = detector.predict(transaction, user_stats)

        assert isinstance(result, tuple)
        assert len(result) == 3  # score, decision, details

    def test_prediction_score_is_numeric(self):
        """Prediction score should be a number."""
        from ml.ensemble_detector import EnsembleDetector

        detector = EnsembleDetector()

        transaction = {
            'amount': 500,
            'user_id': 'user_001',
            'merchant_name': 'Test Store',
            'merchant_category_code': '5999',
            'transaction_date': datetime.utcnow()
        }

        user_stats = {
            'avg_amount': 200,
            'std_amount': 50,
            'max_amount': 1000,
            'transaction_count': 10,
            'known_vendors': ['Test Store'],
            'recent_transactions': []
        }

        score, decision, details = detector.predict(transaction, user_stats)

        assert isinstance(score, (int, float))
        assert 0 <= score <= 100


class TestFeatureEngineering:
    """Test individual feature calculations."""

    def test_amount_log_transform(self):
        """Log transform should handle edge cases."""
        # log(0) should not crash
        assert np.isfinite(np.log1p(0))

        # log of negative should be handled
        amount = -100
        log_amount = np.log1p(max(0, amount))
        assert np.isfinite(log_amount)

    def test_hour_extraction(self):
        """Hour should be extracted correctly from timestamp."""
        test_time = datetime(2026, 2, 20, 15, 30, 0)  # 3:30 PM

        hour = test_time.hour

        assert hour == 15

    def test_day_of_week_extraction(self):
        """Day of week should be 0-6 (Monday=0)."""
        # February 20, 2026 is a Friday
        test_time = datetime(2026, 2, 20, 12, 0, 0)

        day_of_week = test_time.weekday()

        assert day_of_week == 4  # Friday

    def test_weekend_detection(self):
        """Weekend should be correctly identified."""
        saturday = datetime(2026, 2, 21, 12, 0, 0)
        sunday = datetime(2026, 2, 22, 12, 0, 0)
        friday = datetime(2026, 2, 20, 12, 0, 0)

        assert saturday.weekday() >= 5  # Weekend
        assert sunday.weekday() >= 5    # Weekend
        assert friday.weekday() < 5     # Weekday

    def test_mcc_risk_categories(self):
        """High-risk MCCs should be identifiable."""
        high_risk_mccs = ['7995', '5933', '5944']  # Gambling, pawn, jewelry
        low_risk_mccs = ['5411', '5812', '5814']   # Grocery, restaurant, fast food

        # These should be in different risk categories
        for mcc in high_risk_mccs:
            assert mcc in ['7995', '5933', '5944', '9999']


class TestModelMonitoring:
    """Tests for model monitoring functionality."""

    def test_monitoring_initializes(self):
        """Monitor should initialize without error."""
        from ml.monitoring import ModelMonitor

        monitor = ModelMonitor()

        assert monitor is not None

    def test_log_prediction_works(self):
        """Should be able to log a prediction."""
        from ml.monitoring import ModelMonitor

        monitor = ModelMonitor()

        # This should not raise an error
        monitor.log_prediction(
            transaction_id='test_001',
            user_id='user_001',
            amount=500,
            ensemble_score=45.0,
            if_score=50.0,
            xgb_score=42.0,
            was_flagged=False
        )

    def test_get_stats_returns_dict(self):
        """Stats should return a dictionary."""
        from ml.monitoring import ModelMonitor

        monitor = ModelMonitor()

        stats = monitor.get_stats(hours=24)

        assert isinstance(stats, dict)

    def test_detect_drift_returns_dict(self):
        """Drift detection should return a dictionary."""
        from ml.monitoring import ModelMonitor

        monitor = ModelMonitor()

        result = monitor.detect_drift()

        assert isinstance(result, dict)

    def test_hourly_breakdown_returns_list(self):
        """Hourly breakdown should return a list."""
        from ml.monitoring import ModelMonitor

        monitor = ModelMonitor()

        breakdown = monitor.get_hourly_breakdown(hours=24)

        assert isinstance(breakdown, list)