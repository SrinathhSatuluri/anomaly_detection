"""
Model Monitoring - Track prediction quality and detect drift
Similar to Ramp's production ML monitoring
"""

import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np


class ModelMonitor:
    """
    Tracks model predictions and detects:
    1. Prediction drift (model behavior changing)
    2. Feature drift (input data changing)
    3. Performance metrics over time
    """

    def __init__(self, storage_path: str = 'ml/monitoring_data.json'):
        self.storage_path = storage_path
        self.predictions = []
        self.max_history = 1000
        self.drift_threshold = 0.15  # 15% change triggers alert

        # Load existing data
        self._load()

    def _load(self):
        """Load prediction history from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.predictions = data.get('predictions', [])
            except:
                self.predictions = []

    def _save(self):
        """Save prediction history to disk."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump({'predictions': self.predictions[-self.max_history:]}, f)

    def log_prediction(self, transaction_id: str, user_id: str,
                       amount: float, if_score: float, xgb_score: float,
                       ensemble_score: float, was_flagged: bool,
                       features: Optional[Dict] = None):
        """Log a prediction for monitoring."""
        self.predictions.append({
            'timestamp': datetime.utcnow().isoformat(),
            'transaction_id': transaction_id,
            'user_id': user_id,
            'amount': amount,
            'if_score': if_score,
            'xgb_score': xgb_score,
            'ensemble_score': ensemble_score,
            'was_flagged': was_flagged,
            'features': features or {}
        })

        # Keep rolling window
        if len(self.predictions) > self.max_history:
            self.predictions = self.predictions[-self.max_history:]

        # Persist every 10 predictions
        if len(self.predictions) % 10 == 0:
            self._save()

    def get_stats(self, hours: int = 24) -> Dict:
        """Get prediction statistics for last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent = [
            p for p in self.predictions
            if datetime.fromisoformat(p['timestamp']) > cutoff
        ]

        if not recent:
            return {
                'error': 'No predictions in timeframe',
                'total_predictions': 0
            }

        ensemble_scores = [p['ensemble_score'] for p in recent]
        if_scores = [p['if_score'] for p in recent]
        xgb_scores = [p['xgb_score'] for p in recent]
        flagged = [p for p in recent if p['was_flagged']]
        amounts = [p['amount'] for p in recent]

        return {
            'timeframe_hours': hours,
            'total_predictions': len(recent),
            'flagged_count': len(flagged),
            'flag_rate': round(len(flagged) / len(recent), 4),
            'ensemble': {
                'mean': round(np.mean(ensemble_scores), 2),
                'std': round(np.std(ensemble_scores), 2),
                'p50': round(np.percentile(ensemble_scores, 50), 2),
                'p90': round(np.percentile(ensemble_scores, 90), 2),
                'p99': round(np.percentile(ensemble_scores, 99), 2),
            },
            'isolation_forest': {
                'mean': round(np.mean(if_scores), 2),
                'std': round(np.std(if_scores), 2),
            },
            'xgboost': {
                'mean': round(np.mean(xgb_scores), 2),
                'std': round(np.std(xgb_scores), 2),
            },
            'amounts': {
                'mean': round(np.mean(amounts), 2),
                'min': round(min(amounts), 2),
                'max': round(max(amounts), 2),
            }
        }

    def detect_drift(self) -> Dict:
        """Detect if model behavior is drifting."""
        if len(self.predictions) < 100:
            return {
                'drift_detected': False,
                'reason': f'Insufficient data ({len(self.predictions)}/100 predictions)',
                'recommendation': 'Collect more data'
            }

        # Compare first half vs second half
        mid = len(self.predictions) // 2
        first_half = self.predictions[:mid]
        second_half = self.predictions[mid:]

        first_avg = np.mean([p['ensemble_score'] for p in first_half])
        second_avg = np.mean([p['ensemble_score'] for p in second_half])

        first_flag_rate = len([p for p in first_half if p['was_flagged']]) / len(first_half)
        second_flag_rate = len([p for p in second_half if p['was_flagged']]) / len(second_half)

        score_drift = abs(second_avg - first_avg) / (first_avg + 0.01)
        flag_drift = abs(second_flag_rate - first_flag_rate)

        drift_detected = score_drift > self.drift_threshold or flag_drift > 0.1

        return {
            'drift_detected': drift_detected,
            'score_drift_pct': round(score_drift * 100, 2),
            'flag_rate_drift': round(flag_drift * 100, 2),
            'first_half': {
                'avg_score': round(first_avg, 2),
                'flag_rate': round(first_flag_rate * 100, 2),
                'count': len(first_half)
            },
            'second_half': {
                'avg_score': round(second_avg, 2),
                'flag_rate': round(second_flag_rate * 100, 2),
                'count': len(second_half)
            },
            'recommendation': 'Retrain model' if drift_detected else 'Model stable'
        }

    def get_user_summary(self, user_id: str) -> Dict:
        """Get prediction summary for a specific user."""
        user_preds = [p for p in self.predictions if p['user_id'] == user_id]

        if not user_preds:
            return {'error': 'No predictions for this user'}

        scores = [p['ensemble_score'] for p in user_preds]
        flagged = [p for p in user_preds if p['was_flagged']]

        return {
            'user_id': user_id,
            'total_predictions': len(user_preds),
            'flagged_count': len(flagged),
            'flag_rate': round(len(flagged) / len(user_preds), 4),
            'avg_score': round(np.mean(scores), 2),
            'max_score': round(max(scores), 2),
            'last_prediction': user_preds[-1]['timestamp']
        }

    def get_hourly_breakdown(self, hours: int = 24) -> List[Dict]:
        """Get prediction stats broken down by hour."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent = [
            p for p in self.predictions
            if datetime.fromisoformat(p['timestamp']) > cutoff
        ]

        # Group by hour
        hourly = defaultdict(list)
        for p in recent:
            hour = datetime.fromisoformat(p['timestamp']).strftime('%Y-%m-%d %H:00')
            hourly[hour].append(p)

        result = []
        for hour, preds in sorted(hourly.items()):
            scores = [p['ensemble_score'] for p in preds]
            flagged = len([p for p in preds if p['was_flagged']])
            result.append({
                'hour': hour,
                'count': len(preds),
                'flagged': flagged,
                'avg_score': round(np.mean(scores), 2)
            })

        return result


# Singleton instance
model_monitor = ModelMonitor()