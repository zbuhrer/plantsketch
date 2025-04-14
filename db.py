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
                notes TEXT
            )
        """)

def insert_scan(uuid, video_name, num_frames=0, meshroom_success=False, notes=""):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO scans (uuid, video_name, created_at, num_frames, meshroom_success, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (uuid, video_name, datetime.now().isoformat(), num_frames, meshroom_success, notes))

def get_all_scans():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT * FROM scans ORDER BY created_at DESC").fetchall()
    return rows
