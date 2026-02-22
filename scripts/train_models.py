"""
Train fraud detection models (Isolation Forest + XGBoost)
Production-grade training without Metaflow for Windows compatibility

Features the same capabilities as Ramp's ML pipeline:
- Multi-model ensemble (Isolation Forest + XGBoost)
- 30 engineered features (enhanced from 15)
- Model versioning with timestamps
- Metrics tracking (precision, recall, F1, AUC)
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
from sqlalchemy import create_engine


def load_transactions():
    """Load transactions from PostgreSQL."""
    print("📊 Loading transactions from database...")

    database_url = os.getenv(
        'DATABASE_URL',
        'postgresql://postgres:password@127.0.0.1:5432/anomaly_db'
    )

    engine = create_engine(database_url)

    query = """
        SELECT 
            id, user_id, amount, merchant_name,
            merchant_category_code, transaction_date,
            is_flagged, anomaly_score, created_at
        FROM transactions
        WHERE amount IS NOT NULL
    """

    df = pd.read_sql(query, engine)
    print(f"   Loaded {len(df)} transactions")
    return df


def engineer_features(df):
    """Engineer 30 features for ML models."""
    print("🔧 Engineering features...")

    df = df.copy()
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['created_at'] = pd.to_datetime(df['created_at'])

    # === ORIGINAL 15 FEATURES ===

    # Time features
    df['hour'] = df['transaction_date'].dt.hour
    df['day_of_week'] = df['transaction_date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_late_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)

    # Amount features
    df['amount_log'] = np.log1p(df['amount'])
    df['is_round_number'] = ((df['amount'] > 100) & (df['amount'] % 100 == 0)).astype(int)

    # User aggregations
    user_stats = df.groupby('user_id').agg({
        'amount': ['count', 'mean', 'std', 'max', 'median'],
        'merchant_name': 'nunique',
        'created_at': ['min', 'max']
    }).reset_index()
    user_stats.columns = [
        'user_id', 'user_txn_count', 'user_avg_amount', 'user_std_amount',
        'user_max_amount', 'user_median_amount', 'user_vendor_count',
        'user_first_txn', 'user_last_txn'
    ]
    user_stats['user_std_amount'] = user_stats['user_std_amount'].fillna(user_stats['user_avg_amount'] * 0.5)

    df = df.merge(user_stats, on='user_id', how='left')

    df['amount_zscore'] = (df['amount'] - df['user_avg_amount']) / df['user_std_amount'].clip(lower=1)

    # Vendor features
    vendor_counts = df['merchant_name'].value_counts().to_dict()
    df['vendor_frequency'] = df['merchant_name'].map(vendor_counts).fillna(0)
    df['is_new_vendor'] = (~df.duplicated(subset=['user_id', 'merchant_name'], keep='first')).astype(int)

    # MCC risk score
    mcc_risk = {
        '7995': 100, '5944': 80, '5933': 70,
        '5813': 40, '7011': 30, '5814': 10, '5812': 10,
    }
    df['mcc_risk_score'] = df['merchant_category_code'].map(mcc_risk).fillna(0)

    # Time since last transaction
    df = df.sort_values(['user_id', 'created_at'])
    df['time_since_last'] = df.groupby('user_id')['created_at'].diff().dt.total_seconds() / 60
    df['time_since_last'] = df['time_since_last'].fillna(9999)

    # === NEW 15 FEATURES ===

    # 16. Account age (days since first transaction)
    df['account_age_days'] = (df['created_at'] - df['user_first_txn']).dt.total_seconds() / 86400
    df['account_age_days'] = df['account_age_days'].fillna(0)

    # 17. Amount vs user max ratio
    df['amount_vs_max_ratio'] = df['amount'] / df['user_max_amount'].clip(lower=1)

    # 18. Amount vs median ratio (more robust than mean)
    df['amount_vs_median_ratio'] = df['amount'] / df['user_median_amount'].clip(lower=1)

    # 19. Is first of month (payroll fraud pattern)
    df['is_first_of_month'] = (df['transaction_date'].dt.day <= 5).astype(int)

    # 20. Is end of quarter (budget dump pattern)
    df['is_end_of_quarter'] = (
        (df['transaction_date'].dt.month.isin([3, 6, 9, 12])) &
        (df['transaction_date'].dt.day >= 25)
    ).astype(int)

    # 21. Hour buckets (more granular time patterns)
    df['hour_bucket'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3]).astype(int)

    # 22. Transaction velocity (count in last 24h per user) - approximated
    df['txn_rank_in_day'] = df.groupby([df['user_id'], df['transaction_date'].dt.date]).cumcount() + 1

    # 23. Vendor diversity ratio
    df['vendor_diversity'] = df['user_vendor_count'] / df['user_txn_count'].clip(lower=1)

    # 24. Is high value (top 10% globally)
    amount_90th = df['amount'].quantile(0.9)
    df['is_high_value'] = (df['amount'] >= amount_90th).astype(int)

    # 25. Amount percentile within user
    df['amount_user_percentile'] = df.groupby('user_id')['amount'].rank(pct=True)

    # 26. Same vendor same day count
    df['same_vendor_today'] = df.groupby([
        df['user_id'],
        df['merchant_name'],
        df['transaction_date'].dt.date
    ]).cumcount()

    # 27. Is travel-related MCC
    travel_mccs = ['3000', '3001', '3002', '3003', '4511', '7011', '7512']
    df['is_travel'] = df['merchant_category_code'].isin(travel_mccs).astype(int)

    # 28. Is entertainment MCC
    entertainment_mccs = ['5812', '5813', '5814', '7832', '7841', '7911']
    df['is_entertainment'] = df['merchant_category_code'].isin(entertainment_mccs).astype(int)

    # 29. Log of time since last (handles outliers better)
    df['time_since_last_log'] = np.log1p(df['time_since_last'])

    # 30. Rapid succession flag (< 5 min since last)
    df['is_rapid_succession'] = (df['time_since_last'] < 5).astype(int)

    # === FEATURE COLUMNS ===
    feature_columns = [
        # Original 15
        'amount', 'amount_log', 'hour', 'day_of_week', 'is_weekend',
        'is_late_night', 'is_round_number', 'user_txn_count', 'user_avg_amount',
        'user_std_amount', 'amount_zscore', 'vendor_frequency', 'is_new_vendor',
        'mcc_risk_score', 'time_since_last',
        # New 15
        'account_age_days', 'amount_vs_max_ratio', 'amount_vs_median_ratio',
        'is_first_of_month', 'is_end_of_quarter', 'hour_bucket', 'txn_rank_in_day',
        'vendor_diversity', 'is_high_value', 'amount_user_percentile',
        'same_vendor_today', 'is_travel', 'is_entertainment',
        'time_since_last_log', 'is_rapid_succession'
    ]

    X = df[feature_columns].fillna(0).values
    y = df['is_flagged'].fillna(False).astype(int).values

    print(f"   Created {len(feature_columns)} features")
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Positive class ratio: {y.mean():.2%}")

    return X, y, feature_columns


def train_isolation_forest(X_train, X_test, y_test):
    """Train Isolation Forest model."""
    print("\n🌲 Training Isolation Forest...")

    model = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_estimators=100,
        n_jobs=-1
    )
    model.fit(X_train)

    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred == -1).astype(int)

    metrics = {
        'precision': float(precision_score(y_test, y_pred_binary, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_binary, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred_binary, zero_division=0))
    }

    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   F1 Score: {metrics['f1']:.3f}")

    return model, metrics


def train_xgboost(X_train, X_test, y_train, y_test, feature_columns):
    """Train XGBoost model."""
    print("\n🚀 Training XGBoost...")

    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_test, y_proba)) if y_test.sum() > 0 else 0.0
    }

    # Feature importance
    feature_importance = dict(zip(feature_columns, [float(x) for x in model.feature_importances_]))

    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   F1 Score: {metrics['f1']:.3f}")
    print(f"   AUC-ROC: {metrics['auc_roc']:.3f}")

    # Top 5 features
    top_features = sorted(feature_importance.items(), key=lambda x: -x[1])[:5]
    print(f"   Top features: {[f[0] for f in top_features]}")

    return model, metrics, feature_importance


def save_models(if_model, xgb_model, if_metrics, xgb_metrics, feature_columns, feature_importance):
    """Save models with versioning."""
    print("\n💾 Saving models...")

    models_dir = 'ml/models'
    os.makedirs(models_dir, exist_ok=True)

    # Version based on timestamp
    version = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save Isolation Forest
    if_path = f'{models_dir}/isolation_forest_{version}.joblib'
    joblib.dump(if_model, if_path)
    print(f"   Saved: {if_path}")

    # Save XGBoost
    xgb_path = f'{models_dir}/xgboost_{version}.joblib'
    joblib.dump(xgb_model, xgb_path)
    print(f"   Saved: {xgb_path}")

    # Determine primary model
    primary_model = 'xgboost' if xgb_metrics['f1'] >= if_metrics['f1'] else 'isolation_forest'

    # Save metadata
    metadata = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'feature_columns': feature_columns,
        'primary_model': primary_model,
        'metrics': {
            'isolation_forest': if_metrics,
            'xgboost': xgb_metrics
        },
        'feature_importance': feature_importance
    }

    meta_path = f'{models_dir}/metadata_{version}.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Saved: {meta_path}")

    # Save as "latest"
    joblib.dump(if_model, f'{models_dir}/isolation_forest_latest.joblib')
    joblib.dump(xgb_model, f'{models_dir}/xgboost_latest.joblib')
    with open(f'{models_dir}/metadata_latest.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Updated 'latest' models")

    return version, primary_model


def main():
    print("=" * 60)
    print("🧠 Fraud Detection ML Training Pipeline")
    print("=" * 60)
    print(f"   Timestamp: {datetime.now().isoformat()}")
    print()

    # Load data
    df = load_transactions()

    if len(df) < 10:
        print("\n❌ Not enough data to train models (need at least 10 transactions)")
        print("   Run: python -m scripts.mock_data")
        return

    # Engineer features
    X, y, feature_columns = engineer_features(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n📈 Train/Test Split: {len(X_train)} train, {len(X_test)} test")

    # Train models
    if_model, if_metrics = train_isolation_forest(X_train, X_test, y_test)
    xgb_model, xgb_metrics, feature_importance = train_xgboost(
        X_train, X_test, y_train, y_test, feature_columns
    )

    # Compare models
    print("\n" + "=" * 60)
    print("📊 Model Comparison:")
    print(f"   {'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"   {'-' * 50}")
    print(
        f"   {'Isolation Forest':<20} {if_metrics['precision']:>10.3f} {if_metrics['recall']:>10.3f} {if_metrics['f1']:>10.3f}")
    print(
        f"   {'XGBoost':<20} {xgb_metrics['precision']:>10.3f} {xgb_metrics['recall']:>10.3f} {xgb_metrics['f1']:>10.3f}")

    # Save models
    version, primary_model = save_models(
        if_model, xgb_model, if_metrics, xgb_metrics,
        feature_columns, feature_importance
    )

    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print(f"   Version: {version}")
    print(f"   Primary Model: {primary_model}")
    print(f"   Models saved to: ml/models/")
    print("\n   Restart Flask to use the new models:")
    print("   python -m api.app")
    print("=" * 60)


if __name__ == '__main__':
    main()