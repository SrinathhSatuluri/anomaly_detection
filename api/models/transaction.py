import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, Boolean, JSON, Integer
from sqlalchemy.dialects.postgresql import UUID
from api.models import Base


class Transaction(Base):
    """
    Stores transaction data from Ramp API
    Schema matches Ramp's API response format
    """
    __tablename__ = "transactions"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Ramp transaction data
    ramp_transaction_id = Column(String(100), unique=True, index=True, nullable=False)
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default="USD")
    merchant_name = Column(String(255))
    merchant_category_code = Column(String(10))
    card_id = Column(String(100), index=True)
    user_id = Column(String(100), index=True)
    department = Column(String(100))
    memo = Column(String(500))
    state = Column(String(50))  # PENDING, CLEARED, DECLINED, etc.

    # Timestamps
    transaction_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Anomaly detection results
    anomaly_score = Column(Float, default=0.0)
    is_flagged = Column(Boolean, default=False)
    anomaly_reasons = Column(JSON, default=list)
    analyzed_at = Column(DateTime)

    def __repr__(self):
        return f"<Transaction {self.ramp_transaction_id}: ${self.amount} at {self.merchant_name}>"

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "id": str(self.id),
            "ramp_transaction_id": self.ramp_transaction_id,
            "amount": self.amount,
            "currency": self.currency,
            "merchant_name": self.merchant_name,
            "merchant_category_code": self.merchant_category_code,
            "card_id": self.card_id,
            "user_id": self.user_id,
            "department": self.department,
            "memo": self.memo,
            "state": self.state,
            "transaction_date": self.transaction_date.isoformat() if self.transaction_date else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "anomaly_score": self.anomaly_score,
            "is_flagged": self.is_flagged,
            "anomaly_reasons": self.anomaly_reasons,
            "analyzed_at": self.analyzed_at.isoformat() if self.analyzed_at else None,
        }