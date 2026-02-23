"""
Pytest configuration and shared fixtures.
"""
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def app():
    """Create application for testing."""
    from api.app import create_app

    app = create_app()
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'

    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def mock_db():
    """Create mock database session."""
    from unittest.mock import MagicMock

    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = []
    db.query.return_value.filter.return_value.count.return_value = 0
    db.query.return_value.filter.return_value.first.return_value = None

    return db


@pytest.fixture
def sample_transaction():
    """Create sample transaction for testing."""
    from datetime import datetime

    return {
        'id': 'test_txn_001',
        'amount': 1000.00,  # High amount, round number
        'user_id': 'user_eng_001',
        'merchant_name': 'Suspicious LLC',  # New vendor
        'merchant_category_code': '5944',  # Jewelry (high risk)
        'department': 'Engineering',
        'transaction_date': datetime(2026, 2, 20, 3, 0, 0).isoformat()  # 3 AM
    }


@pytest.fixture
def blocked_transaction():
    """Create transaction that should be blocked."""
    from datetime import datetime

    return {
        'id': 'test_txn_blocked',
        'amount': 15000.00,  # Over limit
        'user_id': 'user_eng_001',
        'merchant_name': 'Big Purchase',
        'merchant_category_code': '5999',
        'transaction_date': datetime.utcnow().isoformat()
    }