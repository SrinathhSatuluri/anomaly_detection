"""
Risk Scorer - Calculates anomaly score in real-time
Soft flags that don't block but require review
"""

import math
from datetime import datetime, time
from typing import Dict, List, Tuple

from api.services.ml_detector import ml_detector


class RiskScorer:
    """
    Calculates risk score (0-100) based on multiple signals.
    Higher score = more suspicious.
    """

    # Weight for each signal (must sum to 1.0)
    WEIGHTS = {
        "amount_deviation": 0.25,
        "new_vendor": 0.20,
        "unusual_time": 0.15,
        "round_number": 0.10,
        "velocity": 0.15,
        "category_mismatch": 0.05,
        "ml_score": 0.05,
        "peer_comparison": 0.05,
    }

    def __init__(self):
        self.high_risk_threshold = 70
        self.critical_threshold = 90

    def calculate_score(
            self,
            transaction: Dict,
            user_stats: Dict,
            recent_transactions: List[Dict],
            peer_stats: Dict = None
    ) -> Tuple[float, List[Dict]]:
        """
        Calculate overall risk score.

        Args:
            transaction: Current transaction
            user_stats: User's historical stats (avg_amount, std_amount, typical_categories)
            recent_transactions: User's transactions in last 24 hours

        Returns:
            (score 0-100, list of contributing factors)
        """
        factors = []

        # Signal 1: Amount deviation from user average
        amount_score, amount_reason = self._score_amount_deviation(
            transaction.get("amount", 0),
            user_stats.get("avg_amount", 100),
            user_stats.get("std_amount", 50)
        )
        if amount_score > 0:
            factors.append({
                "signal": "amount_deviation",
                "score": amount_score,
                "reason": amount_reason,
                "weight": self.WEIGHTS["amount_deviation"]
            })

        # Signal 2: New vendor with high amount
        vendor_score, vendor_reason = self._score_new_vendor(
            transaction.get("merchant_name"),
            transaction.get("amount", 0),
            user_stats.get("known_vendors", [])
        )
        if vendor_score > 0:
            factors.append({
                "signal": "new_vendor",
                "score": vendor_score,
                "reason": vendor_reason,
                "weight": self.WEIGHTS["new_vendor"]
            })

        # Signal 3: Unusual time
        time_score, time_reason = self._score_unusual_time(
            transaction.get("transaction_date")
        )
        if time_score > 0:
            factors.append({
                "signal": "unusual_time",
                "score": time_score,
                "reason": time_reason,
                "weight": self.WEIGHTS["unusual_time"]
            })

        # Signal 4: Suspiciously round number
        round_score, round_reason = self._score_round_number(
            transaction.get("amount", 0)
        )
        if round_score > 0:
            factors.append({
                "signal": "round_number",
                "score": round_score,
                "reason": round_reason,
                "weight": self.WEIGHTS["round_number"]
            })

        # Signal 5: Velocity (many transactions in short time)
        velocity_score, velocity_reason = self._score_velocity(
            recent_transactions
        )
        if velocity_score > 0:
            factors.append({
                "signal": "velocity",
                "score": velocity_score,
                "reason": velocity_reason,
                "weight": self.WEIGHTS["velocity"]
            })

        # Signal 6: Category mismatch (unusual category for this user/department)
        category_score, category_reason = self._score_category_mismatch(
            transaction.get("merchant_category_code"),
            transaction.get("department"),
            user_stats.get("typical_categories", [])
        )
        if category_score > 0:
            factors.append({
                "signal": "category_mismatch",
                "score": category_score,
                "reason": category_reason,
                "weight": self.WEIGHTS["category_mismatch"]
            })

        # Signal 7: ML-based anomaly detection
        ml_score_val, ml_reason = self._score_ml_anomaly(
            transaction.get("user_id", ""),
            transaction,
            user_stats
        )
        if ml_score_val > 0:
            factors.append({
                "signal": "ml_score",
                "score": ml_score_val,
                "reason": ml_reason,
                "weight": self.WEIGHTS["ml_score"]
            })

        # Signal 8: Peer comparison (department-based)
        peer_score, peer_reason = self._score_peer_comparison(
            transaction,
            peer_stats or {}
        )
        if peer_score > 0:
            factors.append({
                "signal": "peer_comparison",
                "score": peer_score,
                "reason": peer_reason,
                "weight": self.WEIGHTS["peer_comparison"]
            })

        # Calculate weighted score
        total_score = sum(f["score"] * f["weight"] for f in factors)

        # Normalize to 0-100
        final_score = min(100, total_score)

        return final_score, factors

    def _score_amount_deviation(self, amount: float, avg: float, std: float) -> Tuple[float, str]:
        """Score based on z-score deviation from user average."""
        if std == 0:
            std = avg * 0.5  # Default to 50% if no history

        z_score = (amount - avg) / std if std > 0 else 0

        if z_score <= 2:
            return 0, ""
        elif z_score <= 3:
            return 50, f"Amount ${amount} is {z_score:.1f}x std dev above average (${avg:.0f})"
        elif z_score <= 4:
            return 75, f"Amount ${amount} is {z_score:.1f}x std dev above average (${avg:.0f})"
        else:
            return 100, f"Amount ${amount} is {z_score:.1f}x std dev above average (${avg:.0f})"

    def _score_new_vendor(self, merchant: str, amount: float, known_vendors: List[str]) -> Tuple[float, str]:
        """Score new vendors, especially with high amounts."""
        if not merchant:
            return 0, ""

        merchant_lower = merchant.lower()
        is_known = any(v.lower() in merchant_lower or merchant_lower in v.lower()
                       for v in known_vendors)

        if is_known:
            return 0, ""

        # New vendor - score based on amount
        if amount < 100:
            return 20, f"New vendor '{merchant}' (low amount)"
        elif amount < 500:
            return 50, f"New vendor '{merchant}' with ${amount}"
        elif amount < 1000:
            return 75, f"New vendor '{merchant}' with high amount ${amount}"
        else:
            return 100, f"New vendor '{merchant}' with very high amount ${amount}"

    def _score_unusual_time(self, transaction_date) -> Tuple[float, str]:
        """Score transactions outside business hours."""
        if not transaction_date:
            return 0, ""

        if isinstance(transaction_date, str):
            transaction_date = datetime.fromisoformat(transaction_date.replace("Z", "+00:00"))

        hour = transaction_date.hour
        day = transaction_date.weekday()

        # Weekend
        if day >= 5:
            return 50, f"Weekend transaction ({transaction_date.strftime('%A')})"

        # Late night (10pm - 6am)
        if hour < 6 or hour >= 22:
            return 75, f"Late night transaction ({hour}:00)"

        # Early morning or late evening
        if hour < 8 or hour >= 20:
            return 25, f"Off-hours transaction ({hour}:00)"

        return 0, ""

    def _score_round_number(self, amount: float) -> Tuple[float, str]:
        """Score suspiciously round numbers (fraud indicator)."""
        if amount < 500:
            return 0, ""

        # Check if perfectly round (ends in 00)
        if amount % 100 == 0:
            if amount >= 1000:
                return 75, f"Suspiciously round amount: ${amount}"
            return 50, f"Round amount: ${amount}"

        # Check if ends in 0
        if amount % 10 == 0 and amount >= 1000:
            return 25, f"Round amount: ${amount}"

        return 0, ""

    def _score_velocity(self, recent_transactions: List[Dict]) -> Tuple[float, str]:
        """Score based on transaction velocity (many in short time)."""
        if not recent_transactions:
            return 0, ""

        # Count transactions in last hour
        count = len(recent_transactions)

        if count >= 10:
            return 100, f"{count} transactions in last hour"
        elif count >= 5:
            return 75, f"{count} transactions in last hour"
        elif count >= 3:
            return 50, f"{count} transactions in last hour"

        return 0, ""

    def _score_category_mismatch(self, mcc: str, department: str, typical_categories: List[str]) -> Tuple[float, str]:
        """Score unusual merchant categories for user/department."""
        if not mcc or not typical_categories:
            return 0, ""

        # MCC categories that are unusual for most business users
        suspicious_mccs = {
            "5944": "Jewelry stores",
            "5945": "Hobby/toy/game shops",
            "7832": "Movie theaters",
            "5813": "Bars/taverns",
            "5814": "Fast food",
            "7011": "Hotels/lodging",
        }

        if mcc not in typical_categories and mcc in suspicious_mccs:
            return 60, f"Unusual category: {suspicious_mccs[mcc]} (MCC {mcc})"

        return 0, ""

    def _score_ml_anomaly(self, user_id: str, transaction: Dict, user_stats: Dict) -> Tuple[float, str]:
        """Score using ensemble ML models (Isolation Forest + XGBoost)."""
        try:
            from ml.ensemble_detector import ensemble_detector

            # Add user stats needed for feature extraction
            enhanced_stats = {
                'avg_amount': user_stats.get('avg_amount', 100),
                'std_amount': user_stats.get('std_amount', 50),
                'total_transactions': user_stats.get('total_transactions', 0),
                'known_vendors': user_stats.get('known_vendors', []),
                'vendor_frequency': 1,
                'minutes_since_last': 60
            }

            score, severity, details = ensemble_detector.predict(transaction, enhanced_stats)

            if not details.get('model_loaded'):
                # Fall back to simple Isolation Forest
                from api.services.ml_detector import ml_detector
                return ml_detector.predict(user_id, transaction)[:2]

            score = float(score)

            if score >= 40:
                if_info = details.get('isolation_forest', {})
                xgb_info = details.get('xgboost', {})
                return score, f"ML ensemble anomaly (IF: {if_info.get('score', 0):.0f}, XGB: {xgb_info.get('score', 0):.0f})"

            return 0, ""

        except Exception as e:
            print(f"Ensemble ML error: {e}")
            # Fall back to simple ML detector
            try:
                from api.services.ml_detector import ml_detector
                ml_score, severity, details = ml_detector.predict(user_id, transaction)

                if not details.get("model_exists"):
                    return 0, ""

                ml_score = float(ml_score)

                if ml_score >= 40:
                    return ml_score, f"ML anomaly detected (score: {ml_score:.0f})"

                return 0, ""
            except:
                return 0, ""

    def _score_peer_comparison(self, transaction: Dict, peer_stats: Dict) -> Tuple[float, str]:
        """Score transaction compared to department peers."""
        if not peer_stats:
            return 0, ""

        amount = transaction.get("amount", 0)
        dept_avg = peer_stats.get("dept_avg_amount", 0)
        dept_max = peer_stats.get("dept_max_amount", 0)
        dept_p90 = peer_stats.get("dept_p90_amount", 0)

        if dept_avg == 0:
            return 0, ""

        # Compare against peer statistics
        if amount > dept_max:
            return 100, f"Amount ${amount} exceeds department maximum (${dept_max})"
        elif amount > dept_p90:
            ratio = amount / dept_p90
            return 75, f"Amount ${amount} is {ratio:.1f}x department 90th percentile (${dept_p90})"
        elif amount > dept_avg * 3:
            ratio = amount / dept_avg
            return 50, f"Amount ${amount} is {ratio:.1f}x department average (${dept_avg})"
        elif amount > dept_avg * 2:
            ratio = amount / dept_avg
            return 25, f"Amount ${amount} is {ratio:.1f}x department average (${dept_avg})"

        return 0, ""

    def get_severity(self, score: float) -> str:
        """Convert score to severity level."""
        if score >= self.critical_threshold:
            return "critical"
        elif score >= self.high_risk_threshold:
            return "high"
        elif score >= 50:
            return "medium"
        elif score > 0:
            return "low"
        return "none"


# Singleton instance
risk_scorer = RiskScorer()