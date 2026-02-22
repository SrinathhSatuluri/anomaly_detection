"""
Initialize the database - creates all tables
Run with: python -m scripts.init_db
"""

import sys

sys.path.insert(0, ".")

from api.models import Base, engine, init_db
from api.models.transaction import Transaction
from api.models.anomaly import Alert, AnomalyRule


def main():
    print("Creating database tables...")
    init_db()
    print("Tables created successfully!")

    print("\nTables in database:")
    for table in Base.metadata.tables:
        print(f"  - {table}")


if __name__ == "__main__":
    main()