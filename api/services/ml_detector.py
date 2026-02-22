"""
ML-based Anomaly Detection using Isolation Forest
Unsupervised learning - no labeled fraud data needed
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest


class MLDetector:
    """
    Isolation Forest-based anomaly detector.

    How it works:
    - Isolation Forest isolates anomalies by randomly selecting features
      and split values. Anomalies are easier to isolate, so they have
      shorter path lengths in the tree.
    - Score of -1 = anomaly, +1 = normal
    - We convert to 0-100 scale for consistency with rule-based scoring
    """

    def __init__(self):
        # Model per user (in production, these would be cached/stored)
        self.user_models: Dict[str, IsolationForest] = {}
        self.user_training_data: Dict[str, List[List[float]]] = {}

        # Minimum transactions needed to train a model
        self.min_training_samples = 5

        # Contamination = expected proportion of anomalies
        self.contamination = 0.1  # Expect ~10% anomalies

    def extract_features(self, transaction: Dict) -> List[float]:
        """
        Extract numerical features from transaction for ML model.

        Features:
        1. amount - transaction amount
        2. hour - hour of day (0-23)
        3. day_of_week - day (0=Monday, 6=Sunday)
        4. is_round_number - 1 if amount ends in 00, else 0
        5. amount_log - log of amount (handles scale)
        """
        amount = float(transaction.get("amount", 0))

        # Parse transaction date
        txn_date = transaction.get("transaction_date")
        if isinstance(txn_date, str):
            try:
                txn_date = datetime.fromisoformat(txn_date.replace("Z", "+00:00"))
            except:
                txn_date = datetime.now()
        elif not txn_date:
            txn_date = datetime.now()

        hour = txn_date.hour
        day_of_week = txn_date.weekday()
        is_round = 1.0 if amount > 0 and amount % 100 == 0 else 0.0
        amount_log = np.log1p(amount)  # log(1 + amount) to handle 0

        return [amount, hour, day_of_week, is_round, amount_log]

    def train_user_model(self, user_id: str, transactions: List[Dict]) -> bool:
        """
        Train an Isolation Forest model for a specific user.

        Args:
            user_id: User identifier
            transactions: List of user's historical transactions

        Returns:
            True if model was trained, False if not enough data
        """
        if len(transactions) < self.min_training_samples:
            return False

        # Extract features from all transactions
        features = [self.extract_features(t) for t in transactions]
        X = np.array(features)

        # Store training data for incremental updates
        self.user_training_data[user_id] = features

        # Train Isolation Forest
        model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        model.fit(X)

        self.user_models[user_id] = model
        return True

    def add_transaction_to_training(self, user_id: str, transaction: Dict):
        """Add a new transaction to user's training data."""
        features = self.extract_features(transaction)

        if user_id not in self.user_training_data:
            self.user_training_data[user_id] = []

        self.user_training_data[user_id].append(features)

        # Retrain if we have enough data and model doesn't exist
        if (user_id not in self.user_models and
                len(self.user_training_data[user_id]) >= self.min_training_samples):
            self._retrain_model(user_id)

    def _retrain_model(self, user_id: str):
        """Retrain model with accumulated data."""
        if user_id not in self.user_training_data:
            return

        X = np.array(self.user_training_data[user_id])

        model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        model.fit(X)
        self.user_models[user_id] = model

    def predict(self, user_id: str, transaction: Dict) -> Tuple[float, str, Dict]:
        """
        Predict if transaction is anomalous.

        Args:
            user_id: User identifier
            transaction: Transaction to evaluate

        Returns:
            (score 0-100, severity, details)
            - score: 0 = normal, 100 = highly anomalous
            - severity: none/low/medium/high/critical
            - details: feature values and raw model output
        """
        features = self.extract_features(transaction)

        # If no model for user, return neutral score
        if user_id not in self.user_models:
            return 0, "none", {
                "reason": "Insufficient history for ML model",
                "features": features,
                "model_exists": False
            }

        model = self.user_models[user_id]
        X = np.array([features])

        # Get anomaly score (-1 to 1, where -1 is most anomalous)
        raw_score = model.decision_function(X)[0]

        # Get prediction (-1 = anomaly, 1 = normal)
        prediction = model.predict(X)[0]

        # Convert to 0-100 scale
        # decision_function typically ranges from -0.5 to 0.5
        # We map: -0.5 -> 100 (anomaly), 0.5 -> 0 (normal)
        normalized_score = max(0, min(100, (0.5 - raw_score) * 100))

        # Determine severity
        if normalized_score >= 80:
            severity = "critical"
        elif normalized_score >= 60:
            severity = "high"
        elif normalized_score >= 40:
            severity = "medium"
        elif normalized_score >= 20:
            severity = "low"
        else:
            severity = "none"

        details = {
            "model_exists": True,
            "raw_score": float(raw_score),
            "prediction": "anomaly" if prediction == -1 else "normal",
            "features": {
                "amount": features[0],
                "hour": features[1],
                "day_of_week": features[2],
                "is_round_number": features[3],
                "amount_log": features[4]
            },
            "training_samples": len(self.user_training_data.get(user_id, []))
        }

        return float(normalized_score), severity, details

    def get_model_stats(self, user_id: str) -> Dict:
        """Get statistics about user's ML model."""
        has_model = user_id in self.user_models
        training_count = len(self.user_training_data.get(user_id, []))

        return {
            "user_id": user_id,
            "has_model": has_model,
            "training_samples": training_count,
            "min_required": self.min_training_samples,
            "ready": has_model
        }

    def train_from_database(self, db_session):
        """Train models for all users from existing database transactions."""
        from api.models.transaction import Transaction
        from sqlalchemy import func

        # Get all users with 5+ transactions
        user_counts = db_session.query(
            Transaction.user_id,
            func.count(Transaction.id).label('count')
        ).group_by(Transaction.user_id).having(func.count(Transaction.id) >= self.min_training_samples).all()

        trained = 0
        for user_id, count in user_counts:
            # Get user's transactions
            transactions = db_session.query(Transaction).filter(
                Transaction.user_id == user_id
            ).all()

            # Convert to dicts
            txn_dicts = [t.to_dict() for t in transactions]

            # Train model
            if self.train_user_model(user_id, txn_dicts):
                trained += 1

        print(f"ML Training: Trained models for {trained} users from database")
        return trained


# Singleton instance
ml_detector = MLDetector()