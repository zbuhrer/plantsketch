# db.py
import sqlite3
from datetime import datetime

DB_PATH = "database.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                uuid TEXT PRIMARY KEY,
                video_name TEXT,
                created_at TEXT,
                num_frames INTEGER,
                meshroom_success BOOLEAN,
                notes TEXT,
                input_type TEXT
            )
        """)

        # Check if input_type column exists, add it if it doesn't
        try:
            conn.execute("SELECT input_type FROM scans LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE scans ADD COLUMN input_type TEXT DEFAULT 'video'")

def insert_scan(uuid, video_name, num_frames=0, meshroom_success=False, notes="", input_type="video"):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO scans (uuid, video_name, created_at, num_frames, meshroom_success, notes, input_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (uuid, video_name, datetime.now().isoformat(), num_frames, meshroom_success, notes, input_type))

def get_all_scans():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT * FROM scans ORDER BY created_at DESC").fetchall()
    return rows
