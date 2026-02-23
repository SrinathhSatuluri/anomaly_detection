"""
Flask Application - Expense Anomaly Detection API
"""

from flask import Flask, jsonify
from flask_cors import CORS

from api.config import Config
from api.routes.webhooks import webhooks_bp
from api.routes.alerts import alerts_bp
from api.routes.investigation import investigation_bp


def create_app():
    """Application factory."""
    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(app)

    # Register blueprints
    app.register_blueprint(webhooks_bp, url_prefix="/api/webhooks")
    app.register_blueprint(alerts_bp, url_prefix="/api/alerts")
    app.register_blueprint(investigation_bp, url_prefix="/api")

    # Train ML models from existing data on startup
    with app.app_context():
        try:
            from api.models import SessionLocal
            from api.services.ml_detector import ml_detector

            db = SessionLocal()
            ml_detector.train_from_database(db)
            db.close()
        except Exception as e:
            print(f"ML training skipped: {e}")

    @app.route("/health")
    def health():
        return jsonify({
            "status": "healthy",
            "service": "anomaly-detection-api",
            "version": "1.0.0"
        })

    @app.route("/")
    def root():
        return jsonify({
            "service": "Ramp-Style Expense Anomaly Detection",
            "endpoints": {
                "health": "/health",
                "process_transaction": "POST /api/webhooks/transactions",
                "process_receipt": "POST /api/webhooks/receipts",
                "get_alerts": "GET /api/alerts",
                "alert_stats": "GET /api/alerts/stats",
                "acknowledge_alert": "POST /api/alerts/{id}/acknowledge",
                "resolve_alert": "POST /api/alerts/{id}/resolve"
            }
        })

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5000)