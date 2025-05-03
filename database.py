import sqlite3
import datetime
from flask import session

def init_db():
    with sqlite3.connect('history.db') as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                file_type TEXT,
                output_file TEXT,
                classes TEXT,
                timestamp TEXT
            )
        """)

def save_to_history(user_id, file_type, output_file, classes_list):
    with sqlite3.connect('history.db') as conn:
        conn.execute(
            "INSERT INTO history (user_id, file_type, output_file, classes, timestamp) VALUES (?, ?, ?, ?, ?)",
            (
                user_id,
                file_type,
                output_file,
                ", ".join(classes_list),
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        )

def get_session_history():
    if 'user_id' not in session:
        return []
    with sqlite3.connect('history.db') as conn:
        cur = conn.execute(
            "SELECT id, file_type, output_file, classes, timestamp FROM history WHERE user_id=? ORDER BY id DESC",
            (session['user_id'],)
        )
        return [
            {
                'id': row[0],
                'type': row[1],
                'file': row[2],
                'classes': row[3].split(", "),
                'time': row[4]
            }
            for row in cur.fetchall()
        ]

def get_history_item(request_id, user_id):
    with sqlite3.connect('history.db') as conn:
        cur = conn.execute(
            "SELECT file_type, output_file, classes, timestamp FROM history WHERE id=? AND user_id=?",
            (request_id, user_id)
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            'file_type': row[0],
            'output_file': row[1],
            'classes': row[2].split(", "),
            'timestamp': row[3]
        }
