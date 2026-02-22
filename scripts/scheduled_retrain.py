"""
Scheduled model retraining
Checks if retraining is needed based on:
1. Model age (>24 hours)
2. New transaction volume (>100 new transactions)
3. Drift detection

Run manually: python -m scripts.scheduled_retrain
Run via cron (Linux): 0 2 * * * cd /path/to/project && python -m scripts.scheduled_retrain
Run via Task Scheduler (Windows): Create task to run daily
"""

import os
import json
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text


def get_model_age_hours() -> float:
    """Get age of current model in hours."""
    metadata_path = 'ml/models/metadata_latest.json'

    if not os.path.exists(metadata_path):
        print("No model found - training needed")
        return float('inf')

    with open(metadata_path) as f:
        metadata = json.load(f)

    trained_at = datetime.fromisoformat(metadata['timestamp'])
    age_hours = (datetime.now() - trained_at).total_seconds() / 3600

    return age_hours


def get_new_transaction_count(since: datetime) -> int:
    """Count transactions created since last training."""
    database_url = os.getenv(
        'DATABASE_URL',
        'postgresql://postgres:password@127.0.0.1:5432/anomaly_db'
    )

    engine = create_engine(database_url)

    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM transactions WHERE created_at > :cutoff"),
            {'cutoff': since}
        )
        return result.scalar() or 0


def check_drift() -> bool:
    """Check if model drift is detected."""
    try:
        from ml.monitoring import model_monitor
        drift_info = model_monitor.detect_drift()
        return drift_info.get('drift_detected', False)
    except Exception as e:
        print(f"Could not check drift: {e}")
        return False


def should_retrain(
    max_age_hours: int = 24,
    min_new_transactions: int = 100,
    check_for_drift: bool = True
) -> tuple:
    """
    Determine if retraining is needed.

    Returns:
        (should_retrain: bool, reason: str)
    """
    # Check 1: Model age
    age_hours = get_model_age_hours()
    if age_hours == float('inf'):
        return True, "No model exists"

    if age_hours > max_age_hours:
        return True, f"Model is {age_hours:.1f} hours old (max: {max_age_hours})"

    # Check 2: New transaction volume
    metadata_path = 'ml/models/metadata_latest.json'
    with open(metadata_path) as f:
        metadata = json.load(f)

    trained_at = datetime.fromisoformat(metadata['timestamp'])
    new_txn_count = get_new_transaction_count(trained_at)

    if new_txn_count >= min_new_transactions:
        return True, f"{new_txn_count} new transactions since last training (threshold: {min_new_transactions})"

    # Check 3: Drift detection
    if check_for_drift and check_drift():
        return True, "Model drift detected"

    return False, f"Model is fresh ({age_hours:.1f}h old, {new_txn_count} new txns, no drift)"


def retrain():
    """Run the training pipeline."""
    print("\nStarting model retraining...")
    from scripts.train_models import main as train_main
    train_main()


def main():
    """Main entry point for scheduled retraining."""
    print("=" * 60)
    print("Scheduled Model Retraining Check")
    print(f"   Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    needs_retrain, reason = should_retrain()

    if needs_retrain:
        print(f"\nRetraining needed: {reason}")
        retrain()
    else:
        print(f"\nSkipping retraining: {reason}")

    print("\n" + "=" * 60)
    print("Current Model Status:")

    # Show current model info
    metadata_path = 'ml/models/metadata_latest.json'
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

        print(f"   Version: {metadata.get('version')}")
        print(f"   Trained: {metadata.get('timestamp')}")
        print(f"   Primary Model: {metadata.get('primary_model')}")
        print(f"   Features: {len(metadata.get('feature_columns', []))}")

        metrics = metadata.get('metrics', {})
        if 'xgboost' in metrics:
            xgb = metrics['xgboost']
            print(f"   XGBoost F1: {xgb.get('f1', 0):.3f}, AUC: {xgb.get('auc_roc', 0):.3f}")
        if 'isolation_forest' in metrics:
            iso = metrics['isolation_forest']
            print(f"   Isolation Forest F1: {iso.get('f1', 0):.3f}")

    print("=" * 60)


if __name__ == '__main__':
    main()