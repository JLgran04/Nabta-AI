import sqlite3
from datetime import datetime

def create_connection():
    return sqlite3.connect("users.db", check_same_thread=False)

def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password BLOB NOT NULL,
            role TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            scan_type TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            explanation_en TEXT,
            explanation_ar TEXT,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

def get_all_users():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role FROM users")
    users = cursor.fetchall()
    conn.close()
    return users

def delete_user(user_id):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

def save_scan(username, scan_type, prediction, confidence, explanation_en="", explanation_ar=""):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO scan_history (
            username, scan_type, prediction, confidence,
            explanation_en, explanation_ar, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        username,
        scan_type,
        prediction,
        confidence,
        explanation_en,
        explanation_ar,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

def get_user_history(username):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, scan_type, prediction, confidence, created_at
        FROM scan_history
        WHERE username = ?
        ORDER BY id DESC
    """, (username,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def delete_user_history(username):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM scan_history WHERE username = ?", (username,))
    conn.commit()
    conn.close()
