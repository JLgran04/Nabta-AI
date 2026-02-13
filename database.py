import sqlite3

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
    conn.commit()
    conn.close()
