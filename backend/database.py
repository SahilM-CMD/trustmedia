import sqlite3
from datetime import datetime
import os

DB_PATH = r"S:\Python\TrustMediaBackend\DeepfakeDetector\trustmedia.db"


def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Create detections table
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            media_type TEXT NOT NULL,
            fake_score REAL NOT NULL,
            suspicion_level TEXT NOT NULL,
            source_type TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_device TEXT,
            notes TEXT
        )
    ''')

    conn.commit()
    conn.close()
    print(f"✓ Database initialized at {DB_PATH}")


def save_detection(filename: str, media_type: str, fake_score: float,
                   suspicion_level: str, source_type: str, user_device: str = "mobile"):
    """Save detection result to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''
        INSERT INTO detections 
        (filename, media_type, fake_score, suspicion_level, source_type, user_device)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (filename, media_type, fake_score, suspicion_level, source_type, user_device))

    conn.commit()
    conn.close()
    print(f"✓ Detection saved: {filename} ({suspicion_level})")


def get_all_detections(limit: int = 100):
    """Get all detections from database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''
        SELECT id, filename, media_type, fake_score, suspicion_level, timestamp 
        FROM detections 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))

    results = c.fetchall()
    conn.close()
    return results


def get_detection_stats():
    """Get statistics about all detections"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('SELECT COUNT(*) FROM detections')
    total = c.fetchone()[0]

    c.execute('SELECT COUNT(*) FROM detections WHERE suspicion_level = "HIGH_SUSPICION"')
    high_risk = c.fetchone()[0]

    c.execute('SELECT AVG(fake_score) FROM detections')
    avg_score = c.fetchone()[0] or 0

    conn.close()

    return {
        "total_scans": total,
        "high_risk_detections": high_risk,
        "average_fake_score": round(avg_score, 4)
    }


if __name__ == "__main__":
    init_db()
