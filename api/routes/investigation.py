"""
Investigation Routes
"""

from flask import Blueprint, jsonify, request
from datetime import datetime

investigation_bp = Blueprint('investigation', __name__)


@investigation_bp.route('/investigate/demo', methods=['POST'])
def investigate_demo():
    from api.services.investigation import investigation_agent

    data = request.get_json() or {}

    transaction = data.get('transaction', {
        "id": "demo_001",
        "amount": 5000.00,
        "merchant_name": "Unknown Vendor LLC",
        "merchant_category_code": "5999",
        "user_id": "user_demo",
        "department": "Engineering",
        "transaction_date": datetime.utcnow().isoformat(),
        "risk_score": 72.5
    })

    risk_factors = data.get('risk_factors', [
        {"signal": "new_vendor", "reason": "First transaction with this merchant", "score": 80},
        {"signal": "unusual_time", "reason": "Transaction at 2:30 AM", "score": 60},
        {"signal": "amount_deviation", "reason": "3.2x above user average", "score": 75}
    ])

    user_history = data.get('user_history', [
        {"amount": 150.00, "merchant_name": "Office Depot"},
        {"amount": 89.00, "merchant_name": "Amazon"},
        {"amount": 1200.00, "merchant_name": "Dell Technologies"}
    ])

    report = investigation_agent.investigate(
        transaction=transaction,
        user_history=user_history,
        risk_factors=risk_factors,
        peer_stats={"department": "Engineering", "avg_amount": 800.00, "max_amount": 5000.00}
    )

    return jsonify({
        "mode": "demo",
        "transaction": transaction,
        "investigation": report
    })