"""
Alerts API - View and manage fraud alerts
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
from sqlalchemy import desc

from api.models import SessionLocal
from api.models.anomaly import Alert
from api.models.transaction import Transaction

alerts_bp = Blueprint("alerts", __name__)


@alerts_bp.route("", methods=["GET"])
def get_alerts():
    """
    Get all alerts with optional filtering.

    Query params:
        status: pending, acknowledged, resolved, dismissed
        severity: low, medium, high, critical
        limit: number of results (default 50)
    """
    db = SessionLocal()

    try:
        query = db.query(Alert).order_by(desc(Alert.created_at))

        # Filter by status
        status = request.args.get("status")
        if status:
            query = query.filter(Alert.status == status)

        # Filter by severity
        severity = request.args.get("severity")
        if severity:
            query = query.filter(Alert.severity == severity)

        # Limit results
        limit = request.args.get("limit", 50, type=int)
        alerts = query.limit(limit).all()

        # Build response with transaction details
        result = []
        for alert in alerts:
            txn = db.query(Transaction).filter(
                Transaction.id == alert.transaction_id
            ).first()

            result.append({
                "id": str(alert.id),
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "score": alert.score,
                "description": alert.description,
                "status": alert.status,
                "created_at": alert.created_at.isoformat() if alert.created_at else None,
                "transaction": {
                    "id": txn.ramp_transaction_id if txn else None,
                    "amount": txn.amount if txn else None,
                    "merchant": txn.merchant_name if txn else None,
                    "user_id": txn.user_id if txn else None,
                    "date": txn.transaction_date.isoformat() if txn and txn.transaction_date else None
                }
            })

        return jsonify({
            "alerts": result,
            "count": len(result)
        }), 200

    finally:
        db.close()


@alerts_bp.route("/<alert_id>/acknowledge", methods=["POST"])
def acknowledge_alert(alert_id):
    """Mark an alert as acknowledged."""
    db = SessionLocal()

    try:
        alert = db.query(Alert).filter(Alert.id == alert_id).first()

        if not alert:
            return jsonify({"error": "Alert not found"}), 404

        data = request.get_json() or {}

        alert.status = "acknowledged"
        alert.acknowledged_by = data.get("user", "system")
        alert.acknowledged_at = datetime.utcnow()

        db.commit()

        return jsonify({
            "id": str(alert.id),
            "status": "acknowledged",
            "acknowledged_by": alert.acknowledged_by,
            "acknowledged_at": alert.acknowledged_at.isoformat()
        }), 200

    finally:
        db.close()


@alerts_bp.route("/<alert_id>/resolve", methods=["POST"])
def resolve_alert(alert_id):
    """Mark an alert as resolved."""
    db = SessionLocal()

    try:
        alert = db.query(Alert).filter(Alert.id == alert_id).first()

        if not alert:
            return jsonify({"error": "Alert not found"}), 404

        data = request.get_json() or {}

        alert.status = "resolved"
        alert.resolved_by = data.get("user", "system")
        alert.resolved_at = datetime.utcnow()
        alert.resolution_notes = data.get("notes", "")

        db.commit()

        return jsonify({
            "id": str(alert.id),
            "status": "resolved",
            "resolved_by": alert.resolved_by,
            "resolved_at": alert.resolved_at.isoformat()
        }), 200

    finally:
        db.close()


@alerts_bp.route("/stats", methods=["GET"])
def get_stats():
    """Get alert statistics for dashboard."""
    db = SessionLocal()

    try:
        from sqlalchemy import func

        # Count by status
        status_counts = db.query(
            Alert.status,
            func.count(Alert.id)
        ).group_by(Alert.status).all()

        # Count by severity
        severity_counts = db.query(
            Alert.severity,
            func.count(Alert.id)
        ).group_by(Alert.severity).all()

        # Total transactions
        total_txns = db.query(func.count(Transaction.id)).scalar()
        flagged_txns = db.query(func.count(Transaction.id)).filter(
            Transaction.is_flagged == True
        ).scalar()

        return jsonify({
            "by_status": {s: c for s, c in status_counts},
            "by_severity": {s: c for s, c in severity_counts},
            "transactions": {
                "total": total_txns,
                "flagged": flagged_txns,
                "flagged_rate": round(flagged_txns / total_txns * 100, 2) if total_txns > 0 else 0
            }
        }), 200

    finally:
        db.close()