"""
database.py
-------------
Handles all SQLite database operations for the Fraud Detection app.

Every prediction made through the web app is logged here: the transaction
details the user entered, the model's prediction, and the timestamp.

Functions:
- init_db()              -> creates the transactions table if missing
- insert_transaction()   -> saves a new prediction record
- get_all_transactions() -> fetches recent records (for the dashboard)
- get_summary_counts()   -> returns total / fraud / genuine counts
"""

import sqlite3
from datetime import datetime

DB_NAME = "fraud.db"


def get_connection():
    """Creates and returns a connection to the SQLite database."""
    return sqlite3.connect(DB_NAME)


def init_db():
    """Creates the 'transactions' table if it does not already exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL,
            category TEXT,
            gender TEXT,
            city_pop INTEGER,
            age INTEGER,
            trans_hour INTEGER,
            prediction TEXT,
            confidence REAL,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialized successfully.")


def insert_transaction(amount, category, gender, city_pop, age, trans_hour,
                        prediction, confidence):
    """Inserts a new transaction record into the database."""
    conn = get_connection()
    cursor = conn.cursor()

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO transactions
        (amount, category, gender, city_pop, age, trans_hour, prediction, confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (amount, category, gender, city_pop, age, trans_hour,
          prediction, confidence, created_at))

    conn.commit()
    conn.close()


def insert_transactions_bulk(rows):
    """
    Inserts MANY transaction records at once using executemany, which is
    far faster than calling insert_transaction() in a loop (one commit
    instead of hundreds/thousands). Used by the batch-predict feature.

    'rows' should be a list of tuples:
    (amount, category, gender, city_pop, age, trans_hour, prediction, confidence)
    """
    conn = get_connection()
    cursor = conn.cursor()

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows_with_timestamp = [row + (created_at,) for row in rows]

    cursor.executemany("""
        INSERT INTO transactions
        (amount, category, gender, city_pop, age, trans_hour, prediction, confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows_with_timestamp)

    conn.commit()
    conn.close()


def get_all_transactions(limit=10):
    """Fetches the most recent transaction records, newest first."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, amount, category, gender, city_pop, age, trans_hour,
               prediction, confidence, created_at
        FROM transactions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_summary_counts():
    """Returns total / fraud / genuine counts from LIVE app usage so far."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM transactions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transactions WHERE prediction = 'Fraud'")
    fraud = cursor.fetchone()[0]

    conn.close()

    return {"total": total, "fraud": fraud, "genuine": total - fraud}
