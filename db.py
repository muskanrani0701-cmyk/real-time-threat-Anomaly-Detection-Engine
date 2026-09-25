import sqlite3
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import config

def init_db(db_path: str = config.DATABASE_PATH):
    """Initializes the database schema matching SRS Section 3.2."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id      TEXT NOT NULL,
        captured_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
        risk_score     REAL NOT NULL,
        severity       TEXT NOT NULL, 
        knn_distance   REAL,
        cnn_score      REAL
    );
    """)
    conn.commit()
    conn.close()

def insert_event(
    source_id: str,
    risk_score: float,
    severity: str,
    knn_distance: Optional[float] = None,
    cnn_score: Optional[float] = None,
    db_path: str = config.DATABASE_PATH
) -> int:
    """Inserts an anomaly event into the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO events (source_id, captured_at, risk_score, severity, knn_distance, cnn_score)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        source_id,
        datetime.now(timezone.utc).isoformat(),
        float(risk_score),
        severity,
        float(knn_distance) if knn_distance is not None else None,
        float(cnn_score) if cnn_score is not None else None
    ))
    event_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return event_id

def get_recent_events(limit: int = 50, db_path: str = config.DATABASE_PATH) -> List[Dict[str, Any]]:
    """Retrieves recent anomaly events ordered by time descending."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
    SELECT event_id, source_id, captured_at, risk_score, severity, knn_distance, cnn_score
    FROM events
    ORDER BY event_id DESC
    LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    results = [dict(row) for row in rows]
    conn.close()
    return results

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
