"""
Investigation Routes
Endpoints for AI-powered fraud investigation

This extends your detection system to include investigation automation,
similar to what Finic builds for fraud ops teams.
"""

from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta

investigation_bp = Blueprint('investigation', __name__)


@investigation_bp.route('/investigate/<transaction_id>', methods=['POST'])
def investigate_transaction(transaction_id):
    """
    Generate an AI-powered investigation report for a flagged transaction.
    
    This demonstrates the investigation workflow:
    1. Fetch transaction details
    2. Pull user history
    3. Gather peer statistics
    4. Generate AI analysis
    5. Return actionable report
    
    POST /api/investigate/{transaction_id}
    
    Returns:
        Investigation report with recommendation
    """
    from api.models import SessionLocal
    from api.models.transaction import Transaction
    from api.services.investigation import investigation_agent
    from sqlalchemy import func
    
    db = SessionLocal()
    
    try:
        # 1. Fetch the transaction
        transaction = db.query(Transaction).filter(
            Transaction.id == transaction_id
        ).first()
        
        if not transaction:
            # Try by ramp_transaction_id
            transaction = db.query(Transaction).filter(
                Transaction.ramp_transaction_id == transaction_id
            ).first()
        
        if not transaction:
            return jsonify({
                "error": "Transaction not found",
                "transaction_id": transaction_id
            }), 404
        
        txn_dict = transaction.to_dict()
        
        # 2. Pull user history (last 30 days, up to 50 transactions)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        user_history = db.query(Transaction).filter(
            Transaction.user_id == transaction.user_id,
            Transaction.id != transaction.id,
            Transaction.created_at >= thirty_days_ago
        ).order_by(Transaction.created_at.desc()).limit(50).all()
        
        user_history_dicts = [t.to_dict() for t in user_history]
        
        # 3. Calculate peer statistics (same department)
        peer_stats = None
        if transaction.department:
            peer_data = db.query(
                func.avg(Transaction.amount).label('avg_amount'),
                func.max(Transaction.amount).label('max_amount'),
                func.count(Transaction.id).label('count')
            ).filter(
                Transaction.department == transaction.department,
                Transaction.created_at >= thirty_days_ago
            ).first()
            
            if peer_data and peer_data.count > 0:
                peer_stats = {
                    "department": transaction.department,
                    "avg_amount": float(peer_data.avg_amount or 0),
                    "max_amount": float(peer_data.max_amount or 0),
                    "transaction_count": peer_data.count
                }
        
        # 4. Extract risk factors from stored anomaly_reasons
        risk_factors = []
        if transaction.anomaly_reasons:
            reasons = transaction.anomaly_reasons
            if isinstance(reasons, str):
                import json
                try:
                    reasons = json.loads(reasons)
                except:
                    reasons = [{"signal": "unknown", "reason": reasons}]
            
            if isinstance(reasons, list):
                for r in reasons:
                    if isinstance(r, dict):
                        risk_factors.append(r)
                    else:
                        risk_factors.append({"signal": "flag", "reason": str(r), "score": 50})
        
        # Add risk score to transaction dict
        txn_dict['risk_score'] = transaction.anomaly_score or 0
        
        # 5. Generate investigation report
        report = investigation_agent.investigate(
            transaction=txn_dict,
            user_history=user_history_dicts,
            risk_factors=risk_factors,
            peer_stats=peer_stats
        )
        
        return jsonify({
            "transaction_id": transaction_id,
            "investigation": report,
            "context": {
                "user_history_count": len(user_history_dicts),
                "risk_factors_count": len(risk_factors),
                "peer_comparison_available": peer_stats is not None
            }
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "transaction_id": transaction_id
        }), 500
        
    finally:
        db.close()


@investigation_bp.route('/investigate/demo', methods=['POST'])
def investigate_demo():
    """
    Demo endpoint that generates an investigation report from provided data.
    No database required - useful for demonstrations.
    
    POST /api/investigate/demo
    Body: {
        "transaction": {...},
        "risk_factors": [...],
        "user_history": [...] (optional)
    }
    """
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
        {"amount": 1200.00, "merchant_name": "Dell Technologies"},
        {"amount": 45.00, "merchant_name": "Starbucks"},
        {"amount": 500.00, "merchant_name": "Software License Co"}
    ])
    
    # Calculate average for context
    if user_history:
        avg_amount = sum(t.get('amount', 0) for t in user_history) / len(user_history)
        transaction['user_avg_amount'] = avg_amount
    
    report = investigation_agent.investigate(
        transaction=transaction,
        user_history=user_history,
        risk_factors=risk_factors,
        peer_stats={
            "department": transaction.get("department", "Unknown"),
            "avg_amount": 800.00,
            "max_amount": 5000.00
        }
    )
    
    return jsonify({
        "mode": "demo",
        "transaction": transaction,
        "investigation": report,
        "context": {
            "risk_factors": risk_factors,
            "user_history_count": len(user_history)
        }
    })


@investigation_bp.route('/investigate/batch', methods=['POST'])
def investigate_batch():
    """
    Batch investigate multiple flagged transactions.
    
    POST /api/investigate/batch
    Body: {
        "transaction_ids": ["id1", "id2", ...]
    }
    
    Returns summary of all investigations.
    """
    from api.models import SessionLocal
    from api.models.transaction import Transaction
    from api.services.investigation import investigation_agent
    
    data = request.get_json() or {}
    transaction_ids = data.get('transaction_ids', [])
    
    if not transaction_ids:
        return jsonify({"error": "No transaction_ids provided"}), 400
    
    if len(transaction_ids) > 10:
        return jsonify({"error": "Maximum 10 transactions per batch"}), 400
    
    db = SessionLocal()
    results = []
    
    try:
        for txn_id in transaction_ids:
            transaction = db.query(Transaction).filter(
                Transaction.id == txn_id
            ).first()
            
            if transaction:
                txn_dict = transaction.to_dict()
                txn_dict['risk_score'] = transaction.anomaly_score or 0
                
                # Quick investigation without full history
                report = investigation_agent.investigate(
                    transaction=txn_dict,
                    user_history=[],
                    risk_factors=[],
                    peer_stats=None
                )
                
                results.append({
                    "transaction_id": txn_id,
                    "recommendation": report.get('recommendation'),
                    "risk_level": report.get('risk_level'),
                    "status": "complete"
                })
            else:
                results.append({
                    "transaction_id": txn_id,
                    "status": "not_found"
                })
        
        # Summary statistics
        recommendations = [r.get('recommendation') for r in results if r.get('status') == 'complete']
        
        return jsonify({
            "batch_size": len(transaction_ids),
            "completed": len([r for r in results if r.get('status') == 'complete']),
            "summary": {
                "block": recommendations.count('BLOCK'),
                "escalate": recommendations.count('ESCALATE'),
                "approve": recommendations.count('APPROVE'),
                "review": recommendations.count('REVIEW')
            },
            "results": results
        })
        
    finally:
        db.close()
