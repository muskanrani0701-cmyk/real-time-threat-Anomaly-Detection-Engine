import sqlite3
import json
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import config

def init_db(db_path: str = config.DATABASE_PATH):
    """Initializes the database schema matching SRS Section 3.2 and Biometric extensions."""
    try:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Primary anomaly events table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id      TEXT NOT NULL,
        captured_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
        risk_score     REAL NOT NULL,
        severity       TEXT NOT NULL, 
        knn_distance   REAL,
        cnn_score      REAL,
        face_status    TEXT DEFAULT 'NO_FACE',
        person_name    TEXT,
        face_similarity REAL,
        snapshot_path  TEXT
    );
    """)

    # Check and add newly added columns if table already existed without them
    cursor.execute("PRAGMA table_info(events)")
    existing_columns = [row[1] for row in cursor.fetchall()]
    new_cols = [
        ("face_status", "TEXT DEFAULT 'NO_FACE'"),
        ("person_name", "TEXT"),
        ("face_similarity", "REAL"),
        ("snapshot_path", "TEXT")
    ]
    for col_name, col_type in new_cols:
        if col_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE events ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError:
                pass

    # 2. Authorized personnel profiles with 512-D facial embeddings
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS authorized_personnel (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name      TEXT NOT NULL,
        role_title     TEXT DEFAULT 'Resident / Staff',
        image_path     TEXT NOT NULL,
        embedding_json TEXT NOT NULL,
        created_at     DATETIME DEFAULT CURRENT_TIMESTAMP
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
    face_status: Optional[str] = "NO_FACE",
    person_name: Optional[str] = None,
    face_similarity: Optional[float] = None,
    snapshot_path: Optional[str] = None,
    db_path: str = config.DATABASE_PATH
) -> int:
    """Inserts an anomaly or intruder event into the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO events (
        source_id, captured_at, risk_score, severity, knn_distance, cnn_score,
        face_status, person_name, face_similarity, snapshot_path
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        source_id,
        datetime.now(timezone.utc).isoformat(),
        float(risk_score),
        severity,
        float(knn_distance) if knn_distance is not None else None,
        float(cnn_score) if cnn_score is not None else None,
        face_status,
        person_name,
        float(face_similarity) if face_similarity is not None else None,
        snapshot_path
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
    SELECT event_id, source_id, captured_at, risk_score, severity, knn_distance, cnn_score,
           face_status, person_name, face_similarity, snapshot_path
    FROM events
    ORDER BY event_id DESC
    LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    results = [dict(row) for row in rows]
    conn.close()
    return results

def insert_authorized_person(
    full_name: str,
    role_title: str,
    image_path: str,
    embedding: List[float],
    db_path: str = config.DATABASE_PATH
) -> int:
    """Inserts an authorized personnel record with 512-D face embedding."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO authorized_personnel (full_name, role_title, image_path, embedding_json)
    VALUES (?, ?, ?, ?)
    """, (
        full_name.strip(),
        role_title.strip(),
        image_path,
        json.dumps(embedding)
    ))
    person_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return person_id

def get_all_authorized(db_path: str = config.DATABASE_PATH) -> List[Dict[str, Any]]:
    """Retrieves all registered authorized personnel."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, full_name, role_title, image_path, embedding_json, created_at
    FROM authorized_personnel
    ORDER BY id ASC
    """)
    rows = cursor.fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["embedding"] = json.loads(d["embedding_json"])
        results.append(d)
    conn.close()
    return results

def delete_authorized_person(person_id: int, db_path: str = config.DATABASE_PATH) -> bool:
    """Deletes an authorized personnel record and its photo."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT image_path FROM authorized_personnel WHERE id = ?", (person_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False

    image_path = row[0]
    cursor.execute("DELETE FROM authorized_personnel WHERE id = ?", (person_id,))
    conn.commit()
    conn.close()

    if image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)
        except OSError:
            pass
    return True

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully with biometric tables.")
