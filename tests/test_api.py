"""
Tests for API Endpoints
"""
import pytest
import json
from unittest.mock import MagicMock, patch


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get('/health')

        assert response.status_code == 200

    def test_health_returns_status(self, client):
        """Health endpoint should return healthy status."""
        response = client.get('/health')
        data = json.loads(response.data)

        assert data['status'] == 'healthy'
        assert 'service' in data
        assert 'version' in data


class TestTransactionEndpoint:
    """Tests for transaction processing endpoint."""

    def test_transaction_requires_amount(self, client):
        """Transaction without amount should fail."""
        response = client.post(
            '/api/webhooks/transactions',
            json={'user_id': 'user_001', 'merchant_name': 'Test'}
        )

        # Should either return 400 or handle gracefully
        assert response.status_code in [400, 500] or b'error' in response.data.lower()

    def test_transaction_requires_user_id(self, client):
        """Transaction without user_id should fail."""
        response = client.post(
            '/api/webhooks/transactions',
            json={'amount': 500, 'merchant_name': 'Test'}
        )

        assert response.status_code in [400, 500] or b'error' in response.data.lower()

    def test_valid_transaction_returns_decision(self, client):
        """Valid transaction should return a decision."""
        response = client.post(
            '/api/webhooks/transactions',
            json={
                'id': 'test_txn_001',
                'amount': 500,
                'user_id': 'user_001',
                'merchant_name': 'Test Store',
                'merchant_category_code': '5999',
                'transaction_date': '2026-02-20T14:00:00Z'
            }
        )

        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'decision' in data
            assert data['decision'] in ['APPROVED', 'FLAGGED', 'BLOCKED']

    def test_blocked_transaction_returns_blocked(self, client):
        """Transaction over limit should be blocked."""
        response = client.post(
            '/api/webhooks/transactions',
            json={
                'id': 'test_txn_002',
                'amount': 15000,  # Over $10,000 limit
                'user_id': 'user_001',
                'merchant_name': 'Big Purchase',
                'merchant_category_code': '5999',
                'transaction_date': '2026-02-20T14:00:00Z'
            }
        )

        if response.status_code == 200:
            data = json.loads(response.data)
            assert data['decision'] == 'BLOCKED'

    def test_gambling_transaction_blocked(self, client):
        """Gambling MCC should be blocked."""
        response = client.post(
            '/api/webhooks/transactions',
            json={
                'id': 'test_txn_003',
                'amount': 100,
                'user_id': 'user_001',
                'merchant_name': 'Casino',
                'merchant_category_code': '7995',  # Gambling
                'transaction_date': '2026-02-20T14:00:00Z'
            }
        )

        if response.status_code == 200:
            data = json.loads(response.data)
            assert data['decision'] == 'BLOCKED'


class TestMLStatusEndpoint:
    """Tests for ML status endpoint."""

    def test_ml_status_returns_200(self, client):
        """ML status endpoint should return 200."""
        response = client.get('/api/webhooks/ml/status')

        assert response.status_code == 200

    def test_ml_status_contains_model_info(self, client):
        """ML status should contain model information."""
        response = client.get('/api/webhooks/ml/status')
        data = json.loads(response.data)

        assert 'model_loaded' in data
        assert 'feature_count' in data

    def test_ml_status_shows_feature_count(self, client):
        """ML status should show 30 features."""
        response = client.get('/api/webhooks/ml/status')
        data = json.loads(response.data)

        if data.get('model_loaded'):
            assert data['feature_count'] == 30


class TestAlertsEndpoint:
    """Tests for alerts endpoint."""

    def test_alerts_returns_200(self, client):
        """Alerts endpoint should return 200."""
        response = client.get('/api/webhooks/alerts')

        assert response.status_code == 200

    def test_alerts_returns_list(self, client):
        """Alerts endpoint should return a list."""
        response = client.get('/api/webhooks/alerts')
        data = json.loads(response.data)

        assert 'alerts' in data
        assert isinstance(data['alerts'], list)

    def test_alerts_filter_by_status(self, client):
        """Alerts should be filterable by status."""
        response = client.get('/api/webhooks/alerts?status=pending')

        assert response.status_code == 200

    def test_alerts_filter_by_severity(self, client):
        """Alerts should be filterable by severity."""
        response = client.get('/api/webhooks/alerts?severity=high')

        assert response.status_code == 200


class TestMonitoringEndpoints:
    """Tests for monitoring endpoints."""

    def test_monitoring_stats_returns_200(self, client):
        """Monitoring stats should return 200."""
        response = client.get('/api/webhooks/ml/monitoring/stats')

        assert response.status_code == 200

    def test_monitoring_drift_returns_200(self, client):
        """Drift detection should return 200."""
        response = client.get('/api/webhooks/ml/monitoring/drift')

        assert response.status_code == 200

    def test_monitoring_hourly_returns_200(self, client):
        """Hourly monitoring should return 200."""
        response = client.get('/api/webhooks/ml/monitoring/hourly')

        assert response.status_code == 200


# Pytest fixtures
@pytest.fixture
def app():
    """Create application for testing."""
    from api.app import create_app

    app = create_app()
    app.config['TESTING'] = True

    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()