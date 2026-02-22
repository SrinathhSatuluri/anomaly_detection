"""
Fraud Detection ML Pipeline using Metaflow
Inspired by Ramp's production ML infrastructure
"""

from metaflow import FlowSpec, step, Parameter, current
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os


class FraudDetectionFlow(FlowSpec):
    """
    Production ML pipeline for fraud detection.

    Run with: python -m ml.flows.fraud_detection_flow run
    """

    contamination = Parameter(
        'contamination',
        help='Expected proportion of anomalies',
        default=0.1
    )

    xgb_n_estimators = Parameter(
        'xgb_n_estimators',
        help='Number of XGBoost trees',
        default=100
    )

    test_size = Parameter(
        'test_size',
        help='Proportion of data for testing',
        default=0.2
    )

    @step
    def start(self):
        """Initialize the flow and load data from database."""
        print(f"🚀 Starting Fraud Detection Flow")
        print(f"   Run ID: {current.run_id}")
        print(f"   Timestamp: {datetime.now().isoformat()}")

        self.transactions = self._load_transactions()
        print(f"   Loaded {len(self.transactions)} transactions")

        self.next(self.feature_engineering)

    def _load_transactions(self):
        """Load transactions from PostgreSQL."""
        from sqlalchemy import create_engine

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
        return df

    @step
    def feature_engineering(self):
        """Engineer features for ML models."""
        print("🔧 Engineering features...")

        df = self.transactions.copy()

        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['created_at'] = pd.to_datetime(df['created_at'])

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
            'amount': ['count', 'mean', 'std'],
            'merchant_name': 'nunique'
        }).reset_index()
        user_stats.columns = ['user_id', 'user_txn_count', 'user_avg_amount', 'user_std_amount', 'user_vendor_count']
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

        self.feature_columns = [
            'amount', 'amount_log', 'hour', 'day_of_week', 'is_weekend',
            'is_late_night', 'is_round_number', 'user_txn_count', 'user_avg_amount',
            'user_std_amount', 'amount_zscore', 'vendor_frequency', 'is_new_vendor',
            'mcc_risk_score', 'time_since_last'
        ]

        self.X = df[self.feature_columns].fillna(0).values
        self.y = df['is_flagged'].fillna(False).astype(int).values

        print(f"   Created {len(self.feature_columns)} features")
        print(f"   Feature matrix shape: {self.X.shape}")
        print(f"   Positive class ratio: {self.y.mean():.2%}")

        self.next(self.train_isolation_forest, self.train_xgboost)

    @step
    def train_isolation_forest(self):
        """Train Isolation Forest for unsupervised anomaly detection."""
        from sklearn.ensemble import IsolationForest
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import precision_score, recall_score, f1_score

        print("🌲 Training Isolation Forest...")

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=42
        )

        self.if_model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
            n_jobs=-1
        )
        self.if_model.fit(X_train)

        y_pred = self.if_model.predict(X_test)
        y_pred_binary = (y_pred == -1).astype(int)

        self.if_metrics = {
            'precision': precision_score(y_test, y_pred_binary, zero_division=0),
            'recall': recall_score(y_test, y_pred_binary, zero_division=0),
            'f1': f1_score(y_test, y_pred_binary, zero_division=0)
        }

        print(f"   Precision: {self.if_metrics['precision']:.3f}")
        print(f"   Recall: {self.if_metrics['recall']:.3f}")
        print(f"   F1 Score: {self.if_metrics['f1']:.3f}")

        self.next(self.ensemble)

    @step
    def train_xgboost(self):
        """Train XGBoost classifier for supervised fraud detection."""
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

        print("🚀 Training XGBoost...")

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=42
        )

        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=self.xgb_n_estimators,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        self.xgb_model.fit(X_train, y_train)

        y_pred = self.xgb_model.predict(X_test)
        y_proba = self.xgb_model.predict_proba(X_test)[:, 1]

        self.xgb_metrics = {
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_proba) if y_test.sum() > 0 else 0
        }

        self.feature_importance = dict(zip(
            self.feature_columns,
            self.xgb_model.feature_importances_
        ))

        print(f"   Precision: {self.xgb_metrics['precision']:.3f}")
        print(f"   Recall: {self.xgb_metrics['recall']:.3f}")
        print(f"   F1 Score: {self.xgb_metrics['f1']:.3f}")
        print(f"   AUC-ROC: {self.xgb_metrics['auc_roc']:.3f}")

        self.next(self.ensemble)

    @step
    def ensemble(self, inputs):
        """Combine models into ensemble."""
        print("🎯 Creating ensemble...")

        self.if_model = inputs[0].if_model
        self.if_metrics = inputs[0].if_metrics
        self.xgb_model = inputs[1].xgb_model
        self.xgb_metrics = inputs[1].xgb_metrics
        self.feature_columns = inputs[0].feature_columns
        self.feature_importance = inputs[1].feature_importance

        print(f"\n   Model Comparison:")
        print(f"   {'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"   {'-' * 50}")
        print(
            f"   {'Isolation Forest':<20} {self.if_metrics['precision']:>10.3f} {self.if_metrics['recall']:>10.3f} {self.if_metrics['f1']:>10.3f}")
        print(
            f"   {'XGBoost':<20} {self.xgb_metrics['precision']:>10.3f} {self.xgb_metrics['recall']:>10.3f} {self.xgb_metrics['f1']:>10.3f}")

        self.primary_model = 'xgboost' if self.xgb_metrics['f1'] >= self.if_metrics['f1'] else 'isolation_forest'
        print(f"\n   Primary model: {self.primary_model}")

        self.next(self.save_models)

    @step
    def save_models(self):
        """Save trained models with versioning."""
        import joblib

        print("💾 Saving models...")

        models_dir = 'ml/models'
        os.makedirs(models_dir, exist_ok=True)

        version = current.run_id

        joblib.dump(self.if_model, f'{models_dir}/isolation_forest_{version}.joblib')
        joblib.dump(self.xgb_model, f'{models_dir}/xgboost_{version}.joblib')

        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'feature_columns': self.feature_columns,
            'primary_model': self.primary_model,
            'metrics': {
                'isolation_forest': self.if_metrics,
                'xgboost': self.xgb_metrics
            },
            'feature_importance': self.feature_importance
        }

        with open(f'{models_dir}/metadata_{version}.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save as latest
        joblib.dump(self.if_model, f'{models_dir}/isolation_forest_latest.joblib')
        joblib.dump(self.xgb_model, f'{models_dir}/xgboost_latest.joblib')
        with open(f'{models_dir}/metadata_latest.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"   Models saved with version: {version}")

        self.next(self.end)

    @step
    def end(self):
        """Pipeline complete."""
        print("\n✅ Fraud Detection Flow Complete!")
        print(f"   Run ID: {current.run_id}")
        print(f"   Primary Model: {self.primary_model}")


if __name__ == '__main__':
    FraudDetectionFlow()