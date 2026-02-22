import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from api.models import Base


class Alert(Base):
    """
    Stores alerts generated when anomalies are detected
    """
    __tablename__ = "alerts"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Link to transaction
    transaction_id = Column(UUID(as_uuid=True), ForeignKey("transactions.id"), nullable=False)

    # Alert details
    alert_type = Column(String(50), nullable=False)  # unusual_amount, new_vendor, odd_time, etc.
    severity = Column(String(20), nullable=False)  # low, medium, high, critical
    score = Column(Float, nullable=False)
    description = Column(Text)

    # Status tracking
    status = Column(String(20), default="pending")  # pending, acknowledged, resolved, dismissed
    acknowledged_by = Column(String(100))
    acknowledged_at = Column(DateTime)
    resolved_by = Column(String(100))
    resolved_at = Column(DateTime)
    resolution_notes = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Alert {self.alert_type}: {self.severity} - {self.status}>"

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "id": str(self.id),
            "transaction_id": str(self.transaction_id),
            "alert_type": self.alert_type,
            "severity": self.severity,
            "score": self.score,
            "description": self.description,
            "status": self.status,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_notes": self.resolution_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AnomalyRule(Base):
    """
    Configurable rules for anomaly detection
    """
    __tablename__ = "anomaly_rules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Rule definition
    rule_name = Column(String(100), unique=True, nullable=False)
    rule_type = Column(String(50), nullable=False)  # amount_threshold, time_based, velocity, etc.
    description = Column(Text)

    # Rule parameters (JSON for flexibility)
    parameters = Column(String(500))  # e.g., {"threshold": 5000, "std_devs": 3}

    # Status
    is_active = Column(String(5), default="true")

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<AnomalyRule {self.rule_name}: {self.rule_type}>"