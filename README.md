# Expense Anomaly Detection System

A real-time fraud detection system for corporate expense management, inspired by Ramp's transaction monitoring architecture. The system implements a three-layer detection pipeline combining rule-based filters, weighted risk scoring, and machine learning ensemble models.

## Table of Contents

- [System Architecture](#system-architecture)
- [Detection Pipeline](#detection-pipeline)
- [Machine Learning Infrastructure](#machine-learning-infrastructure)
- [API Reference](#api-reference)
- [Database Schema](#database-schema)
- [Deployment](#deployment)
- [Local Development](#local-development)
- [Configuration](#configuration)
- [Testing](#testing)

---

## System Architecture

```
                                    +------------------+
                                    |   Dashboard      |
                                    |   (Netlify)      |
                                    +--------+---------+
                                             |
                                             | HTTPS
                                             v
+------------------+              +----------+-----------+
|                  |   Webhook   |                      |
|  Transaction     +------------>+    Flask API         |
|  Source          |   POST      |    (Render)          |
|                  |             |                      |
+------------------+             +----------+-----------+
                                            |
                    +-----------------------+-----------------------+
                    |                       |                       |
                    v                       v                       v
           +--------+-------+    +---------+--------+    +---------+--------+
           |                |    |                  |    |                  |
           |  Layer 1       |    |  Layer 2         |    |  Layer 3         |
           |  Instant Rules |    |  Risk Scoring    |    |  ML Ensemble     |
           |                |    |                  |    |                  |
           +--------+-------+    +---------+--------+    +---------+--------+
                    |                       |                       |
                    +-----------------------+-----------------------+
                                            |
                                            v
                                 +----------+-----------+
                                 |                      |
                                 |  PostgreSQL          |
                                 |  (Render)            |
                                 |                      |
                                 +----------------------+
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| API Server | Flask 3.0, Gunicorn | Request handling, webhook processing |
| Database | PostgreSQL 15 | Transaction storage, alert management |
| ML Framework | scikit-learn, XGBoost | Anomaly detection models |
| Model Serving | joblib | Model serialization and loading |
| Frontend | HTML, Tailwind CSS, Chart.js | Dashboard and visualization |
| Hosting | Render (API), Netlify (Frontend) | Production deployment |

---

## Detection Pipeline

The system processes transactions through three sequential layers. A transaction may be terminated at any layer or passed through to the next.

### Layer 1: Instant Rules (Hard Blocks)

Deterministic rules that immediately block transactions meeting specific criteria. These rules have zero tolerance and require no scoring.

| Rule | Condition | Action |
|------|-----------|--------|
| OVER_LIMIT | amount > $10,000 | BLOCK |
| BLOCKED_MCC | MCC in [7995, 5933, 9999] | BLOCK |
| DUPLICATE | Same user, amount, merchant within 10 min | BLOCK |
| RECEIPT_MISMATCH | Receipt amount differs by > 10% | BLOCK |

Implementation: `api/services/instant_rules.py`

### Layer 2: Risk Scoring (Soft Flags)

Weighted scoring system that evaluates multiple risk signals. Each signal produces a score from 0-100, which is then multiplied by its weight to contribute to the final risk score.

| Signal | Weight | Description |
|--------|--------|-------------|
| amount_deviation | 0.20 | Standard deviations from user's mean spending |
| new_vendor | 0.20 | First transaction with this merchant |
| unusual_time | 0.15 | Transaction outside normal hours (10pm-6am, weekends) |
| velocity | 0.15 | Number of transactions in rolling 1-hour window |
| round_number | 0.10 | Suspiciously round amounts ($X00, $X000) |
| ml_score | 0.10 | Ensemble model anomaly score |
| peer_comparison | 0.05 | Deviation from department spending norms |
| category_mismatch | 0.05 | Unusual merchant category for user |

Decision thresholds:
- Score >= 70: FLAGGED (requires manual review)
- Score < 70: APPROVED (logged with risk factors)

Implementation: `api/services/risk_scorer.py`

### Layer 3: ML Ensemble

Machine learning models that detect anomalies based on learned patterns from historical transaction data.

#### Model Architecture

The ensemble combines two complementary approaches:

1. **Isolation Forest** (Unsupervised)
   - Detects anomalies without requiring labeled fraud data
   - Effective for identifying novel fraud patterns
   - Weight in ensemble: 0.4

2. **XGBoost Classifier** (Supervised)
   - Learns from historically flagged transactions
   - Better precision on known fraud patterns
   - Weight in ensemble: 0.6

Final ensemble score: `0.4 * IF_score + 0.6 * XGB_score`

Implementation: `ml/ensemble_detector.py`

---

## Machine Learning Infrastructure

### Feature Engineering

The system extracts 30 features from each transaction:

**Temporal Features (5)**
- hour, day_of_week, is_weekend, is_late_night, hour_bucket

**Amount Features (7)**
- amount, amount_log, is_round_number, amount_zscore
- amount_vs_max_ratio, amount_vs_median_ratio, amount_user_percentile

**User Behavioral Features (8)**
- user_txn_count, user_avg_amount, user_std_amount
- account_age_days, vendor_diversity, txn_rank_in_day
- same_vendor_today, is_rapid_succession

**Vendor Features (3)**
- vendor_frequency, is_new_vendor, is_high_value

**Category Features (4)**
- mcc_risk_score, is_travel, is_entertainment, category_first_time

**Time Delta Features (3)**
- time_since_last, time_since_last_log, is_first_of_month, is_end_of_quarter

Implementation: `scripts/train_models.py`

### Model Versioning

Models are stored with timestamp-based versioning:

```
ml/models/
├── isolation_forest_20260220_005134.joblib
├── xgboost_20260220_005134.joblib
├── metadata_20260220_005134.json
├── isolation_forest_latest.joblib -> (symlink)
├── xgboost_latest.joblib -> (symlink)
└── metadata_latest.json -> (symlink)
```

Metadata includes:
- Training timestamp
- Feature columns (ordered)
- Model performance metrics (precision, recall, F1, AUC-ROC)
- Feature importance rankings

### Model Monitoring

The system tracks prediction statistics and detects model drift:

**Metrics Tracked:**
- Prediction volume (hourly, daily)
- Score distribution (mean, std, percentiles)
- Flag rate over time
- Per-model scores (IF vs XGBoost)

**Drift Detection:**
- Compares score distributions between time windows
- Alerts when mean score shifts > 15%
- Recommends retraining when drift is detected

Implementation: `ml/monitoring.py`

### Retraining Pipeline

Automated retraining based on:
- Model age (> 24 hours)
- New transaction volume (> 100 new records)
- Detected model drift

Implementation: `scripts/scheduled_retrain.py`

---

## API Reference

Base URL: `https://fraud-detection-api-as51.onrender.com`

### Transaction Processing

**POST /api/webhooks/transactions**

Process a transaction through the detection pipeline.

Request:
```json
{
  "id": "txn_001",
  "amount": 500.00,
  "user_id": "user_eng_001",
  "merchant_name": "Office Supplies Inc",
  "merchant_category_code": "5943",
  "department": "Engineering",
  "transaction_date": "2026-02-21T14:30:00Z"
}
```

Response:
```json
{
  "transaction_id": "txn_001",
  "decision": "APPROVED",
  "risk_score": 23.5,
  "factors": [
    {
      "signal": "new_vendor",
      "score": 75,
      "reason": "New vendor 'Office Supplies Inc' with moderate amount $500",
      "weight": 0.2
    }
  ],
  "reasons": ["New vendor 'Office Supplies Inc' with moderate amount $500"],
  "alerts_created": 0,
  "processed_in_ms": 145
}
```

### Model Status

**GET /api/webhooks/ml/status**

Returns current model information.

Response:
```json
{
  "model_loaded": true,
  "version": "20260220_005134",
  "trained_at": "2026-02-20T00:51:34.114338",
  "primary_model": "isolation_forest",
  "feature_count": 30,
  "metrics": {
    "isolation_forest": {"precision": 1.0, "recall": 0.333, "f1": 0.5},
    "xgboost": {"precision": 0.5, "recall": 0.333, "f1": 0.4, "auc_roc": 0.958}
  }
}
```

### Feature Importance

**GET /api/webhooks/ml/features**

Returns XGBoost feature importance rankings.

### Monitoring Statistics

**GET /api/webhooks/ml/monitoring/stats?hours=24**

Returns prediction statistics for the specified time window.

**GET /api/webhooks/ml/monitoring/drift**

Returns drift detection analysis.

### Alerts

**GET /api/webhooks/alerts?status=pending&severity=high**

List alerts with optional filtering.

**PUT /api/webhooks/alerts/{id}/status**

Update alert status (pending, reviewed, dismissed).

### Receipt Verification

**POST /api/webhooks/receipts/verify**

Verify receipt image against claimed transaction amount.

Request: multipart/form-data with `receipt` (image file), `amount` (float), `merchant` (string, optional)

---

## Database Schema

### transactions

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| ramp_transaction_id | VARCHAR(100) | External transaction identifier |
| amount | FLOAT | Transaction amount |
| currency | VARCHAR(10) | Currency code (default: USD) |
| merchant_name | VARCHAR(255) | Merchant display name |
| merchant_category_code | VARCHAR(10) | MCC code |
| card_id | VARCHAR(100) | Card identifier |
| user_id | VARCHAR(100) | User identifier |
| department | VARCHAR(100) | User's department |
| memo | VARCHAR(500) | Transaction memo |
| state | VARCHAR(50) | Transaction state |
| transaction_date | TIMESTAMP | When transaction occurred |
| created_at | TIMESTAMP | Record creation time |
| anomaly_score | FLOAT | Calculated risk score |
| is_flagged | BOOLEAN | Whether transaction was flagged |
| anomaly_reasons | JSON | Array of detection reasons |
| analyzed_at | TIMESTAMP | When analysis completed |

Indexes: user_id, card_id, ramp_transaction_id (unique)

### alerts

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| transaction_id | UUID | Foreign key to transactions |
| alert_type | VARCHAR(50) | Type: hard_block, risk_score, receipt_mismatch |
| severity | VARCHAR(20) | critical, high, medium, low |
| score | FLOAT | Risk score at time of alert |
| description | TEXT | Human-readable description |
| status | VARCHAR(20) | pending, reviewed, dismissed |
| acknowledged_by | VARCHAR(100) | User who reviewed |
| acknowledged_at | TIMESTAMP | When reviewed |
| created_at | TIMESTAMP | Alert creation time |

---

## Deployment

### Production URLs

| Service | URL |
|---------|-----|
| API | https://fraud-detection-api-as51.onrender.com |
| Dashboard | https://vocal-fenglisu-61e36a.netlify.app |
| Health Check | https://fraud-detection-api-as51.onrender.com/health |

### Environment Variables

| Variable | Description |
|----------|-------------|
| DATABASE_URL | PostgreSQL connection string |
| SECRET_KEY | Flask secret key |
| FLASK_ENV | Environment (production/development) |
| PYTHON_VERSION | Python version for Render (3.11.7) |

### Deployment Process

**Backend (Render):**
1. Push to main branch triggers automatic deployment
2. Render builds using `pip install -r requirements.txt`
3. Starts with `gunicorn wsgi:app`
4. Database migrations run on first request

**Frontend (Netlify):**
1. Manual deploy via drag-and-drop
2. Static files served from CDN
3. API calls proxied to Render backend

---

## Local Development

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- PostgreSQL client (optional)

### Setup

```bash
# Clone repository
git clone https://github.com/SrinathhSatuluri/anomaly_detection.git
cd anomaly_detection

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Start database and message queue
docker-compose up -d

# Initialize database
python -m scripts.init_db

# Train ML models
python -m scripts.train_models

# Start API server
python -m api.app
```

### Running Tests

```bash
pytest tests/ -v
```

### Generating Mock Data

```bash
python -m scripts.mock_data
```

---

## Configuration

### Risk Scoring Weights

Weights can be adjusted in `api/services/risk_scorer.py`:

```python
WEIGHTS = {
    "amount_deviation": 0.20,
    "new_vendor": 0.20,
    "unusual_time": 0.15,
    "round_number": 0.10,
    "velocity": 0.15,
    "category_mismatch": 0.05,
    "ml_score": 0.10,
    "peer_comparison": 0.05,
}
```

### Instant Rule Thresholds

Configured in `api/services/instant_rules.py`:

```python
MAX_AMOUNT = 10000
BLOCKED_MCCS = ["7995", "5933", "9999"]
DUPLICATE_WINDOW_MINUTES = 10
```

### ML Model Parameters

Training parameters in `scripts/train_models.py`:

```python
# Isolation Forest
contamination = 0.1
n_estimators = 100
random_state = 42

# XGBoost
n_estimators = 100
max_depth = 5
learning_rate = 0.1
scale_pos_weight = (calculated from class imbalance)
```

---

## Project Structure

```
anomaly-detection/
├── api/
│   ├── app.py                 # Flask application factory
│   ├── config.py              # Configuration management
│   ├── models/
│   │   ├── __init__.py        # SQLAlchemy setup
│   │   ├── transaction.py     # Transaction model
│   │   └── anomaly.py         # Alert, AnomalyRule models
│   ├── routes/
│   │   ├── webhooks.py        # Transaction processing endpoints
│   │   └── alerts.py          # Alert management endpoints
│   └── services/
│       ├── instant_rules.py   # Layer 1: Hard block rules
│       ├── risk_scorer.py     # Layer 2: Weighted scoring
│       ├── ml_detector.py     # Legacy ML detector
│       └── receipt_ocr.py     # Receipt verification
├── ml/
│   ├── ensemble_detector.py   # Layer 3: ML ensemble
│   ├── monitoring.py          # Model monitoring
│   ├── models/                # Trained model artifacts
│   └── flows/                 # Training pipelines
├── scripts/
│   ├── init_db.py             # Database initialization
│   ├── train_models.py        # Model training pipeline
│   ├── mock_data.py           # Test data generation
│   └── scheduled_retrain.py   # Automated retraining
├── dashboard/
│   └── index.html             # Frontend application
├── tests/                     # Test suite
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # Local development services
├── wsgi.py                    # WSGI entry point
├── Procfile                   # Render deployment config
└── README.md
```

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Average response time | 100-200ms |
| P99 response time | < 500ms |
| Model inference time | < 50ms |
| Throughput (estimated) | 50-100 req/sec |

Note: Performance measured on Render free tier. Production deployment would show improved latency and throughput.

---

## Limitations and Future Work

**Current Limitations:**
- Single-region deployment (no geographic redundancy)
- Synchronous processing (no async queue for heavy analysis)
- Limited historical data for model training
- No A/B testing infrastructure for model comparison

**Potential Enhancements:**
- Integration with card network risk scores (Visa, Mastercard)
- LLM-powered explanation generation for flagged transactions
- Real-time WebSocket updates for dashboard
- Kubernetes deployment for horizontal scaling
- Feature store for real-time feature computation

---

## References

- [Ramp Engineering Blog: Production ML with Metaflow](https://builders.ramp.com/post/metaflow-production-ml)
- [Ramp 2025 Release Notes](https://ramp.com/blog/2025-release-notes)
- [scikit-learn Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## License

MIT License - See LICENSE file for details.

## Author

Srinath Satuluri - [GitHub](https://github.com/SrinathhSatuluri)
