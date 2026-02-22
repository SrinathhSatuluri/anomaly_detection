"""
Generate realistic mock transaction data for demo
Simulates corporate expense patterns with some anomalies
"""

import random
import requests
from datetime import datetime, timedelta

API_URL = "http://127.0.0.1:5000/api/webhooks/transactions"

# Realistic merchants by category
MERCHANTS = {
    "food": [
        ("Starbucks", "5814"),
        ("Chipotle", "5812"),
        ("Uber Eats", "5812"),
        ("Sweetgreen", "5812"),
        ("Panera Bread", "5812"),
    ],
    "travel": [
        ("United Airlines", "4511"),
        ("Delta Airlines", "4511"),
        ("Marriott Hotels", "7011"),
        ("Hilton Hotels", "7011"),
        ("Uber", "4121"),
        ("Lyft", "4121"),
    ],
    "office": [
        ("Amazon", "5999"),
        ("Staples", "5943"),
        ("Office Depot", "5943"),
        ("Apple Store", "5732"),
    ],
    "software": [
        ("AWS", "7372"),
        ("Google Cloud", "7372"),
        ("Zoom", "7372"),
        ("Slack", "7372"),
        ("GitHub", "7372"),
    ],
    "suspicious": [
        ("Vegas Casino Online", "7995"),
        ("Lucky Slots", "7995"),
        ("Gold & Diamonds LLC", "5944"),
        ("Anonymous Gift Cards", "5999"),
    ]
}

# Typical amounts by category
AMOUNTS = {
    "food": (8, 75),
    "travel": (50, 800),
    "office": (20, 300),
    "software": (10, 500),
    "suspicious": (200, 5000),
}

# User profiles
USERS = [
    {"id": "user_eng_001", "name": "Alice (Engineer)", "department": "Engineering"},
    {"id": "user_eng_002", "name": "Bob (Engineer)", "department": "Engineering"},
    {"id": "user_sales_001", "name": "Carol (Sales)", "department": "Sales"},
    {"id": "user_sales_002", "name": "Dave (Sales)", "department": "Sales"},
    {"id": "user_exec_001", "name": "Eve (Executive)", "department": "Executive"},
]


def generate_normal_transaction(user, txn_id):
    """Generate a normal business transaction."""
    category = random.choices(
        ["food", "travel", "office", "software"],
        weights=[40, 20, 25, 15]
    )[0]

    merchant, mcc = random.choice(MERCHANTS[category])
    min_amt, max_amt = AMOUNTS[category]
    amount = round(random.uniform(min_amt, max_amt), 2)

    # Business hours (8am - 7pm, weekdays)
    days_ago = random.randint(0, 30)
    hour = random.randint(8, 19)
    txn_date = datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23))
    txn_date = txn_date.replace(hour=hour, minute=random.randint(0, 59))

    return {
        "id": f"txn_{txn_id:04d}",
        "amount": amount,
        "user_id": user["id"],
        "merchant_name": merchant,
        "merchant_category_code": mcc,
        "department": user["department"],
        "transaction_date": txn_date.isoformat() + "Z"
    }


def generate_suspicious_transaction(user, txn_id, anomaly_type):
    """Generate a suspicious transaction."""
    txn_date = datetime.now() - timedelta(days=random.randint(0, 7))

    if anomaly_type == "blocked_merchant":
        merchant, mcc = random.choice(MERCHANTS["suspicious"])
        amount = round(random.uniform(200, 2000), 2)

    elif anomaly_type == "high_amount":
        category = random.choice(["food", "office"])
        merchant, mcc = random.choice(MERCHANTS[category])
        amount = round(random.uniform(2000, 8000), 2)

    elif anomaly_type == "round_number":
        merchant, mcc = random.choice(MERCHANTS["office"])
        amount = random.choice([1000, 2000, 3000, 5000])

    elif anomaly_type == "late_night":
        merchant, mcc = random.choice(MERCHANTS["food"])
        amount = round(random.uniform(50, 300), 2)
        txn_date = txn_date.replace(hour=random.randint(23, 23), minute=random.randint(0, 59))

    else:
        merchant, mcc = random.choice(MERCHANTS["office"])
        amount = round(random.uniform(100, 500), 2)

    return {
        "id": f"txn_{txn_id:04d}",
        "amount": amount,
        "user_id": user["id"],
        "merchant_name": merchant,
        "merchant_category_code": mcc,
        "department": user["department"],
        "transaction_date": txn_date.isoformat() + "Z"
    }


def send_transaction(txn):
    """Send transaction to API."""
    try:
        response = requests.post(API_URL, json=txn, timeout=5)
        result = response.json()

        status_icon = {
            "APPROVED": "✅",
            "FLAGGED": "⚠️",
            "BLOCKED": "🚫"
        }.get(result.get("decision"), "❓")

        print(
            f"{status_icon} {txn['id']}: ${txn['amount']:.2f} at {txn['merchant_name'][:20]:20s} -> {result.get('decision')} (score: {result.get('risk_score')})")

        return result
    except Exception as e:
        print(f"❌ Error sending {txn['id']}: {e}")
        return None


def main():
    print("=" * 60)
    print("🏦 Generating Mock Transaction Data")
    print("=" * 60)

    txn_id = random.randint(1000, 9000)
    results = {"APPROVED": 0, "FLAGGED": 0, "BLOCKED": 0}

    # Generate normal transactions (80%)
    print("\n📊 Normal Transactions:")
    print("-" * 40)
    for _ in range(20):
        user = random.choice(USERS)
        txn = generate_normal_transaction(user, txn_id)
        result = send_transaction(txn)
        if result:
            results[result.get("decision", "APPROVED")] += 1
        txn_id += 1

    # Generate suspicious transactions (20%)
    print("\n🔍 Suspicious Transactions:")
    print("-" * 40)

    anomaly_types = ["blocked_merchant", "high_amount", "round_number", "late_night"]
    for anomaly_type in anomaly_types:
        user = random.choice(USERS)
        txn = generate_suspicious_transaction(user, txn_id, anomaly_type)
        result = send_transaction(txn)
        if result:
            results[result.get("decision", "APPROVED")] += 1
        txn_id += 1

    # Summary
    print("\n" + "=" * 60)
    print("📈 Summary:")
    print(f"   ✅ Approved: {results['APPROVED']}")
    print(f"   ⚠️  Flagged:  {results['FLAGGED']}")
    print(f"   🚫 Blocked:  {results['BLOCKED']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
