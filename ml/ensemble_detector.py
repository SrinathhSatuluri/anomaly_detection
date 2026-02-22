"""
Production Ensemble Detector
Combines Isolation Forest + XGBoost like Ramp's multi-model approach
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import joblib
from ml.monitoring import model_monitor


class EnsembleDetector:
    """
    Production-grade ensemble detector using:
    1. Isolation Forest (unsupervised) - catches novel anomalies
    2. XGBoost (supervised) - learns from labeled fraud cases

    Ensemble strategy:
    - Both models vote
    - Final score = weighted average
    - XGBoost weighted higher if we have labeled data
    """

    def __init__(self, models_dir: str = 'ml/models'):
        self.models_dir = models_dir
        self.if_model = None
        self.xgb_model = None
        self.metadata = None
        self.feature_columns = None
        self.is_loaded = False

        # Ensemble weights
        self.if_weight = 0.4
        self.xgb_weight = 0.6

        # Try to load models on init
        self.load_models()

    def load_models(self) -> bool:
        """Load the latest trained models."""
        try:
            if_path = f'{self.models_dir}/isolation_forest_latest.joblib'
            xgb_path = f'{self.models_dir}/xgboost_latest.joblib'
            meta_path = f'{self.models_dir}/metadata_latest.json'

            if not all(os.path.exists(p) for p in [if_path, xgb_path, meta_path]):
                print("ML Models not found. Run the training flow first.")
                return False

            self.if_model = joblib.load(if_path)
            self.xgb_model = joblib.load(xgb_path)

            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)

            self.feature_columns = self.metadata['feature_columns']
            self.is_loaded = True

            print(f"ML Models loaded (version: {self.metadata['version']})")
            return True

        except Exception as e:
            print(f"Failed to load models: {e}")
            return False

    def extract_features(self, transaction: Dict, user_stats: Dict) -> np.ndarray:
        """Extract 30 features matching the training pipeline."""
        amount = float(transaction.get('amount', 0))

        txn_date = transaction.get('transaction_date')
        if isinstance(txn_date, str):
            try:
                txn_date = datetime.fromisoformat(txn_date.replace('Z', '+00:00'))
            except:
                txn_date = datetime.now()
        elif not txn_date:
            txn_date = datetime.now()

        hour = txn_date.hour
        day_of_week = txn_date.weekday()

        # MCC risk scores
        mcc_risk = {
            '7995': 100, '5944': 80, '5933': 70,
            '5813': 40, '7011': 30, '5814': 10, '5812': 10
        }

        # Travel and entertainment MCCs
        travel_mccs = ['3000', '3001', '3002', '3003', '4511', '7011', '7512']
        entertainment_mccs = ['5812', '5813', '5814', '7832', '7841', '7911']

        mcc = transaction.get('merchant_category_code', '')
        user_avg = user_stats.get('avg_amount', 100)
        user_std = user_stats.get('std_amount', 50)
        user_max = user_stats.get('max_amount', user_avg * 2)
        user_median = user_stats.get('median_amount', user_avg)
        user_txn_count = user_stats.get('total_transactions', 1)
        user_vendor_count = user_stats.get('vendor_count', 1)
        minutes_since_last = user_stats.get('minutes_since_last', 9999)
        account_age_days = user_stats.get('account_age_days', 0)

        # Calculate features
        features = {
            # Original 15
            'amount': amount,
            'amount_log': np.log1p(amount),
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'is_late_night': 1 if (hour >= 22 or hour < 6) else 0,
            'is_round_number': 1 if (amount > 100 and amount % 100 == 0) else 0,
            'user_txn_count': user_txn_count,
            'user_avg_amount': user_avg,
            'user_std_amount': user_std,
            'amount_zscore': (amount - user_avg) / max(user_std, 1),
            'vendor_frequency': user_stats.get('vendor_frequency', 1),
            'is_new_vendor': 1 if transaction.get('merchant_name') not in user_stats.get('known_vendors', []) else 0,
            'mcc_risk_score': mcc_risk.get(mcc, 0),
            'time_since_last': minutes_since_last,
            # New 15
            'account_age_days': account_age_days,
            'amount_vs_max_ratio': amount / max(user_max, 1),
            'amount_vs_median_ratio': amount / max(user_median, 1),
            'is_first_of_month': 1 if txn_date.day <= 5 else 0,
            'is_end_of_quarter': 1 if (txn_date.month in [3, 6, 9, 12] and txn_date.day >= 25) else 0,
            'hour_bucket': 0 if hour < 6 else (1 if hour < 12 else (2 if hour < 18 else 3)),
            'txn_rank_in_day': user_stats.get('txn_rank_in_day', 1),
            'vendor_diversity': user_vendor_count / max(user_txn_count, 1),
            'is_high_value': 1 if amount >= user_stats.get('amount_90th', 1000) else 0,
            'amount_user_percentile': user_stats.get('amount_user_percentile', 0.5),
            'same_vendor_today': user_stats.get('same_vendor_today', 0),
            'is_travel': 1 if mcc in travel_mccs else 0,
            'is_entertainment': 1 if mcc in entertainment_mccs else 0,
            'time_since_last_log': np.log1p(minutes_since_last),
            'is_rapid_succession': 1 if minutes_since_last < 5 else 0,
        }

        return np.array([[features.get(col, 0) for col in self.feature_columns]])

    def predict(self, transaction: Dict, user_stats: Dict) -> Tuple[float, str, Dict]:
        """
        Make ensemble prediction.

        Returns:
            (score 0-100, severity, details)
        """
        if not self.is_loaded:
            return 0, 'none', {'error': 'Models not loaded', 'model_loaded': False}

        try:
            # Extract features
            X = self.extract_features(transaction, user_stats)

            # Isolation Forest prediction
            if_raw = self.if_model.decision_function(X)[0]
            if_pred = self.if_model.predict(X)[0]
            # Convert: -0.5 (anomaly) -> 100, 0.5 (normal) -> 0
            if_score = max(0, min(100, (0.5 - if_raw) * 100))

            # XGBoost prediction
            xgb_proba = self.xgb_model.predict_proba(X)[0][1]
            xgb_score = xgb_proba * 100

            # Ensemble score (weighted average)
            ensemble_score = (
                    self.if_weight * if_score +
                    self.xgb_weight * xgb_score
            )

            # Determine severity
            if ensemble_score >= 80:
                severity = 'critical'
            elif ensemble_score >= 60:
                severity = 'high'
            elif ensemble_score >= 40:
                severity = 'medium'
            elif ensemble_score >= 20:
                severity = 'low'
            else:
                severity = 'none'

            # Prediction label
            if ensemble_score >= 50:
                prediction = 'fraud'
            else:
                prediction = 'normal'

            details = {
                'model_loaded': True,
                'model_version': self.metadata.get('version', 'unknown'),
                'ensemble_score': float(ensemble_score),
                'isolation_forest': {
                    'score': float(if_score),
                    'raw_score': float(if_raw),
                    'prediction': 'anomaly' if if_pred == -1 else 'normal'
                },
                'xgboost': {
                    'score': float(xgb_score),
                    'probability': float(xgb_proba)
                },
                'prediction': prediction,
                'weights': {
                    'isolation_forest': self.if_weight,
                    'xgboost': self.xgb_weight
                }
            }

            # Log prediction for monitoring
            try:
                model_monitor.log_prediction(
                    transaction_id=transaction.get('id', 'unknown'),
                    user_id=transaction.get('user_id', 'unknown'),
                    amount=float(transaction.get('amount', 0)),
                    if_score=float(if_score),
                    xgb_score=float(xgb_score),
                    ensemble_score=float(ensemble_score),
                    was_flagged=(ensemble_score >= 40)
                )
            except Exception as e:
                print(f"Monitoring log error: {e}")

            return float(ensemble_score), severity, details

        except Exception as e:
            return 0, 'none', {'error': str(e), 'model_loaded': True}

    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        if not self.is_loaded:
            return {'loaded': False}

        return {
            'loaded': True,
            'version': self.metadata.get('version'),
            'trained_at': self.metadata.get('timestamp'),
            'primary_model': self.metadata.get('primary_model'),
            'metrics': self.metadata.get('metrics'),
            'feature_count': len(self.feature_columns),
            'feature_importance': self.metadata.get('feature_importance', {})
        }


# Singleton instance
ensemble_detector = EnsembleDetector()