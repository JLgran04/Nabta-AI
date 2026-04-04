import sqlite3
from datetime import datetime

DB_NAME = "users.db"

def create_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password BLOB NOT NULL,
            role TEXT NOT NULL
        )
    """)

    # Scan history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            scan_type TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            explanation_en TEXT,
            explanation_ar TEXT,
            image_data BLOB,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

def save_scan(
    username,
    scan_type,
    prediction,
    confidence,
    explanation_en="",
    explanation_ar="",
    image_data=None
):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO scan_history (
            username,
            scan_type,
            prediction,
            confidence,
            explanation_en,
            explanation_ar,
            image_data,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        username,
        scan_type,
        prediction,
        confidence,
        explanation_en,
        explanation_ar,
        image_data,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

def get_user_history(username, scan_type_filter="All"):
    conn = create_connection()
    cursor = conn.cursor()

    if scan_type_filter == "All":
        cursor.execute("""
            SELECT
                id,
                scan_type,
                prediction,
                confidence,
                explanation_en,
                explanation_ar,
                image_data,
                created_at
            FROM scan_history
            WHERE username = ?
            ORDER BY id DESC
        """, (username,))
    else:
        cursor.execute("""
            SELECT
                id,
                scan_type,
                prediction,
                confidence,
                explanation_en,
                explanation_ar,
                image_data,
                created_at
            FROM scan_history
            WHERE username = ? AND scan_type = ?
            ORDER BY id DESC
        """, (username, scan_type_filter))

    rows = cursor.fetchall()
    conn.close()
    return rows

def delete_user_history(username):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM scan_history
        WHERE username = ?
    """, (username,))

    conn.commit()
    conn.close()
