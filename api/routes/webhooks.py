"""
Webhooks - Real-time transaction processing endpoint
"""

import time
from datetime import datetime, timedelta
from typing import Dict
from flask import Blueprint, request, jsonify
from sqlalchemy import and_, func

from api.models import SessionLocal
from api.models.transaction import Transaction
from api.models.anomaly import Alert
from api.services.instant_rules import instant_rules
from api.services.risk_scorer import risk_scorer
from api.services.ml_detector import ml_detector

webhooks_bp = Blueprint("webhooks", __name__)


@webhooks_bp.route("/transactions", methods=["POST"])
def process_transaction():
    """Process incoming transaction webhook."""
    start_time = time.time()

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload"}), 400

    required = ["id", "amount", "user_id"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    db = SessionLocal()

    try:
        # LAYER 1: INSTANT RULES (Hard Blocks)
        should_block, block_reasons = instant_rules.evaluate(data)

        # Check for duplicates
        recent_txns = get_recent_transactions(db, data["user_id"], minutes=10)
        is_duplicate, dup_reason = instant_rules.check_duplicate(data, recent_txns)

        if is_duplicate:
            should_block = True
            block_reasons.append(dup_reason)

        # If blocked, return immediately
        if should_block:
            transaction = store_transaction(db, data, is_flagged=True,
                                            anomaly_score=100, reasons=block_reasons)
            create_alert(db, transaction.id, "hard_block", "critical", 100,
                         "; ".join(block_reasons))
            db.commit()

            elapsed = int((time.time() - start_time) * 1000)
            return jsonify({
                "transaction_id": data["id"],
                "decision": "BLOCKED",
                "risk_score": 100,
                "reasons": block_reasons,
                "alerts_created": 1,
                "processed_in_ms": elapsed
            }), 200

        # LAYER 2: RISK SCORING (Soft Flags)
        user_stats = get_user_stats(db, data["user_id"])
        recent_hour = get_recent_transactions(db, data["user_id"], minutes=60)
        peer_stats = get_peer_stats(db, data.get("department"))

        risk_score, factors = risk_scorer.calculate_score(data, user_stats, recent_hour, peer_stats)
        severity = risk_scorer.get_severity(risk_score)

        # Determine decision
        if risk_score >= 70:
            decision = "FLAGGED"
        else:
            decision = "APPROVED"

        # Store transaction
        reasons = [f["reason"] for f in factors if f["reason"]]
        transaction = store_transaction(
            db, data,
            is_flagged=(decision == "FLAGGED"),
            anomaly_score=risk_score,
            reasons=reasons
        )

        # Create alert if flagged
        alerts_created = 0
        if decision == "FLAGGED":
            alert_desc = "; ".join(reasons) if reasons else "High risk score"
            create_alert(db, transaction.id, "risk_score", severity, risk_score, alert_desc)
            alerts_created = 1

        db.commit()

        # Train ML model with this transaction
        ml_detector.add_transaction_to_training(data["user_id"], data)

        elapsed = int((time.time() - start_time) * 1000)
        return jsonify({
            "transaction_id": data["id"],
            "decision": decision,
            "risk_score": round(risk_score, 2),
            "reasons": reasons,
            "factors": factors,
            "alerts_created": alerts_created,
            "processed_in_ms": elapsed
        }), 200

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500

    finally:
        db.close()


@webhooks_bp.route("/receipts", methods=["POST"])
def process_receipt():
    """Process receipt submission and check for mismatch."""
    start_time = time.time()
    data = request.get_json()

    if not data or "transaction_id" not in data:
        return jsonify({"error": "Missing transaction_id"}), 400

    db = SessionLocal()

    try:
        transaction = db.query(Transaction).filter(
            Transaction.ramp_transaction_id == data["transaction_id"]
        ).first()

        if not transaction:
            return jsonify({"error": "Transaction not found"}), 404

        receipt_amount = data.get("receipt_amount")
        is_mismatch, mismatch_reason = instant_rules.check_receipt_mismatch(
            transaction.amount, receipt_amount
        )

        if is_mismatch:
            transaction.is_flagged = True
            transaction.anomaly_score = max(transaction.anomaly_score or 0, 85)
            existing_reasons = transaction.anomaly_reasons or []
            transaction.anomaly_reasons = existing_reasons + [mismatch_reason]

            create_alert(db, transaction.id, "receipt_mismatch", "high", 85, mismatch_reason)
            db.commit()

            elapsed = int((time.time() - start_time) * 1000)
            return jsonify({
                "transaction_id": data["transaction_id"],
                "status": "MISMATCH_DETECTED",
                "reason": mismatch_reason,
                "action": "BLOCKED_PENDING_REVIEW",
                "processed_in_ms": elapsed
            }), 200

        elapsed = int((time.time() - start_time) * 1000)
        return jsonify({
            "transaction_id": data["transaction_id"],
            "status": "MATCHED",
            "processed_in_ms": elapsed
        }), 200

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500

    finally:
        db.close()


@webhooks_bp.route("/ml/status", methods=["GET"])
def ml_status():
    """Get ML model status and metrics."""
    from ml.ensemble_detector import ensemble_detector

    info = ensemble_detector.get_model_info()

    return jsonify({
        "model_loaded": info.get('loaded', False),
        "version": info.get('version'),
        "trained_at": info.get('trained_at'),
        "primary_model": info.get('primary_model'),
        "metrics": info.get('metrics'),
        "feature_count": info.get('feature_count')
    }), 200


@webhooks_bp.route("/ml/features", methods=["GET"])
def ml_features():
    """Get feature importance from XGBoost."""
    from ml.ensemble_detector import ensemble_detector

    info = ensemble_detector.get_model_info()
    importance = info.get('feature_importance', {})

    # Sort by importance
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    return jsonify({
        "model_loaded": info.get('loaded', False),
        "feature_importance": dict(sorted_features),
        "top_features": sorted_features[:10]  # Top 10 most important
    }), 200


@webhooks_bp.route("/ml/monitoring/stats", methods=["GET"])
def ml_monitoring_stats():
    """Get model monitoring statistics."""
    from ml.monitoring import model_monitor

    hours = request.args.get('hours', 24, type=int)
    stats = model_monitor.get_stats(hours=hours)

    return jsonify(stats), 200


@webhooks_bp.route("/ml/monitoring/drift", methods=["GET"])
def ml_monitoring_drift():
    """Check for model drift."""
    from ml.monitoring import model_monitor

    drift = model_monitor.detect_drift()

    return jsonify(drift), 200


@webhooks_bp.route("/ml/monitoring/hourly", methods=["GET"])
def ml_monitoring_hourly():
    """Get hourly prediction breakdown."""
    from ml.monitoring import model_monitor

    hours = request.args.get('hours', 24, type=int)
    hourly = model_monitor.get_hourly_breakdown(hours=hours)

    return jsonify({"hourly_stats": hourly}), 200


@webhooks_bp.route("/ml/monitoring/user/<user_id>", methods=["GET"])
def ml_monitoring_user(user_id):
    """Get prediction summary for a user."""
    from ml.monitoring import model_monitor

    summary = model_monitor.get_user_summary(user_id)

    return jsonify(summary), 200


@webhooks_bp.route("/transactions", methods=["GET"])
def get_transactions():
    """List all transactions with optional filtering."""
    db = SessionLocal()

    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        flagged_only = request.args.get('flagged', 'false').lower() == 'true'

        query = db.query(Transaction)

        if flagged_only:
            query = query.filter(Transaction.is_flagged == True)

        transactions = query.order_by(Transaction.created_at.desc()).offset(offset).limit(limit).all()

        return jsonify({
            "transactions": [t.to_dict() for t in transactions],
            "total": query.count(),
            "limit": limit,
            "offset": offset
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@webhooks_bp.route("/alerts", methods=["GET"])
def get_alerts():
    """List all alerts with optional filtering."""
    db = SessionLocal()

    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        status = request.args.get('status')  # pending, investigating, resolved
        severity = request.args.get('severity')  # low, medium, high, critical

        query = db.query(Alert)

        if status:
            query = query.filter(Alert.status == status)
        if severity:
            query = query.filter(Alert.severity == severity)

        alerts = query.order_by(Alert.created_at.desc()).offset(offset).limit(limit).all()

        return jsonify({
            "alerts": [a.to_dict() for a in alerts],
            "total": query.count(),
            "limit": limit,
            "offset": offset
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@webhooks_bp.route("/stats/summary", methods=["GET"])
def get_summary_stats():
    """Get summary statistics for dashboard."""
    db = SessionLocal()

    try:
        from datetime import datetime, timedelta
        from sqlalchemy import func

        now = datetime.utcnow()
        last_7d = now - timedelta(days=7)
        prev_7d = now - timedelta(days=14)

        # Total transactions last 7 days
        total_7d = db.query(func.count(Transaction.id)).filter(
            Transaction.created_at >= last_7d
        ).scalar() or 0

        # Total transactions previous 7 days
        total_prev_7d = db.query(func.count(Transaction.id)).filter(
            Transaction.created_at >= prev_7d,
            Transaction.created_at < last_7d
        ).scalar() or 0

        # Flagged last 7 days
        flagged_7d = db.query(func.count(Transaction.id)).filter(
            Transaction.created_at >= last_7d,
            Transaction.is_flagged == True
        ).scalar() or 0

        # Flagged previous 7 days
        flagged_prev_7d = db.query(func.count(Transaction.id)).filter(
            Transaction.created_at >= prev_7d,
            Transaction.created_at < last_7d,
            Transaction.is_flagged == True
        ).scalar() or 0

        # Average risk score
        avg_score = db.query(func.avg(Transaction.anomaly_score)).filter(
            Transaction.created_at >= last_7d
        ).scalar() or 0

        # Pending alerts
        pending_alerts = db.query(func.count(Alert.id)).filter(
            Alert.status == 'pending'
        ).scalar() or 0

        # Calculate changes
        txn_change = ((total_7d - total_prev_7d) / max(total_prev_7d, 1)) * 100
        flagged_change = ((flagged_7d - flagged_prev_7d) / max(flagged_prev_7d, 1)) * 100

        # Daily breakdown for chart
        daily_data = []
        for i in range(7):
            day_start = (now - timedelta(days=6-i)).replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)

            day_count = db.query(func.count(Transaction.id)).filter(
                Transaction.created_at >= day_start,
                Transaction.created_at < day_end
            ).scalar() or 0

            day_flagged = db.query(func.count(Transaction.id)).filter(
                Transaction.created_at >= day_start,
                Transaction.created_at < day_end,
                Transaction.is_flagged == True
            ).scalar() or 0

            daily_data.append({
                'date': day_start.strftime('%m/%d'),
                'day': day_start.strftime('%a'),
                'transactions': day_count,
                'flagged': day_flagged
            })

        return jsonify({
            "total_transactions_7d": total_7d,
            "txn_change_pct": round(txn_change, 1),
            "flagged_7d": flagged_7d,
            "flagged_change_pct": round(flagged_change, 1),
            "avg_risk_score": round(float(avg_score), 1),
            "pending_alerts": pending_alerts,
            "daily_breakdown": daily_data
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


# Helper functions

def store_transaction(db, data, is_flagged, anomaly_score, reasons):
    """Store transaction in database."""
    txn_date = data.get("transaction_date")
    if isinstance(txn_date, str):
        txn_date = datetime.fromisoformat(txn_date.replace("Z", "+00:00"))
    elif not txn_date:
        txn_date = datetime.utcnow()

    transaction = Transaction(
        ramp_transaction_id=data["id"],
        amount=data.get("amount", 0),
        currency=data.get("currency", "USD"),
        merchant_name=data.get("merchant_name"),
        merchant_category_code=data.get("merchant_category_code"),
        card_id=data.get("card_id"),
        user_id=data.get("user_id"),
        department=data.get("department"),
        memo=data.get("memo"),
        state=data.get("state", "PENDING"),
        transaction_date=txn_date,
        is_flagged=is_flagged,
        anomaly_score=anomaly_score,
        anomaly_reasons=reasons,
        analyzed_at=datetime.utcnow()
    )

    db.add(transaction)
    db.flush()
    return transaction


def create_alert(db, transaction_id, alert_type, severity, score, description):
    """Create an alert for a transaction."""
    alert = Alert(
        transaction_id=transaction_id,
        alert_type=alert_type,
        severity=severity,
        score=score,
        description=description,
        status="pending"
    )
    db.add(alert)


def get_recent_transactions(db, user_id, minutes):
    """Get user's recent transactions."""
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)

    transactions = db.query(Transaction).filter(
        and_(
            Transaction.user_id == user_id,
            Transaction.created_at >= cutoff
        )
    ).all()

    return [t.to_dict() for t in transactions]


def get_peer_stats(db, department: str) -> Dict:
    """Get statistics for users in same department."""
    if not department:
        return {}

    stats = db.query(
        func.avg(Transaction.amount).label("avg_amount"),
        func.max(Transaction.amount).label("max_amount"),
        func.percentile_cont(0.9).within_group(Transaction.amount).label("p90_amount")
    ).filter(Transaction.department == department).first()

    if not stats or not stats.avg_amount:
        return {}

    return {
        "dept_avg_amount": float(stats.avg_amount or 100),
        "dept_max_amount": float(stats.max_amount or 1000),
        "dept_p90_amount": float(stats.p90_amount or 500)
    }


def get_user_stats(db, user_id):
    """Get user's historical statistics."""
    stats = db.query(
        func.avg(Transaction.amount).label("avg_amount"),
        func.stddev(Transaction.amount).label("std_amount"),
        func.count(Transaction.id).label("total_transactions")
    ).filter(Transaction.user_id == user_id).first()

    vendors = db.query(Transaction.merchant_name).filter(
        and_(
            Transaction.user_id == user_id,
            Transaction.merchant_name.isnot(None)
        )
    ).distinct().limit(100).all()

    categories = db.query(Transaction.merchant_category_code).filter(
        and_(
            Transaction.user_id == user_id,
            Transaction.merchant_category_code.isnot(None)
        )
    ).distinct().all()

    return {
        "avg_amount": float(stats.avg_amount or 100),
        "std_amount": float(stats.std_amount or 50),
        "total_transactions": stats.total_transactions or 0,
        "known_vendors": [v[0] for v in vendors if v[0]],
        "typical_categories": [c[0] for c in categories if c[0]]
    }


# ============== OCR ROUTES ==============

@webhooks_bp.route("/receipts/ocr/status", methods=["GET"])
def ocr_status():
    """Check if OCR is available."""
    from api.services.receipt_ocr import receipt_ocr

    return jsonify({
        "available": receipt_ocr.is_available(),
        "message": "OCR ready" if receipt_ocr.is_available() else "Install pytesseract and Tesseract-OCR"
    }), 200


@webhooks_bp.route("/receipts/ocr", methods=["POST"])
def ocr_receipt():
    """Extract data from receipt image."""
    from api.services.receipt_ocr import receipt_ocr

    if not receipt_ocr.is_available():
        return jsonify({
            "error": "OCR not available. Install pytesseract and Tesseract-OCR."
        }), 503

    if 'receipt' not in request.files:
        return jsonify({"error": "No receipt file provided"}), 400

    file = request.files['receipt']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    import tempfile
    import os

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    try:
        result = receipt_ocr.extract_from_image(temp_path)
        return jsonify(result), 200
    finally:
        os.remove(temp_path)
        os.rmdir(temp_dir)


@webhooks_bp.route("/receipts/verify", methods=["POST"])
def verify_receipt():
    """Verify receipt matches claimed transaction."""
    from api.services.receipt_ocr import receipt_ocr

    if not receipt_ocr.is_available():
        return jsonify({
            "error": "OCR not available. Install pytesseract and Tesseract-OCR."
        }), 503

    if 'receipt' not in request.files:
        return jsonify({"error": "No receipt file provided"}), 400

    file = request.files['receipt']
    claimed_amount = request.form.get('amount', type=float)
    claimed_merchant = request.form.get('merchant')

    if not claimed_amount:
        return jsonify({"error": "Claimed amount is required"}), 400

    import tempfile
    import os

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    try:
        is_verified, details = receipt_ocr.verify_transaction(
            temp_path,
            claimed_amount,
            claimed_merchant
        )
        return jsonify({"verified": is_verified, "details": details}), 200
    finally:
        os.remove(temp_path)
        os.rmdir(temp_dir)