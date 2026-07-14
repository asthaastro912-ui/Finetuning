"""Tiny SQLite-backed request log shared by the FastAPI server (writer) and
the Streamlit dashboard (reader). SQLite is enough for a single-instance demo
deployment; swap for Postgres if this ever needs to run multi-process.
"""
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    question TEXT NOT NULL,
    context_chars INTEGER NOT NULL,
    prediction TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    numeric_hallucination_rate REAL,
    status TEXT NOT NULL DEFAULT 'ok',
    error TEXT
);
"""


def get_connection(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def log_request(conn, question, context_chars, prediction, latency_ms,
                 numeric_hallucination_rate=None, status="ok", error=None):
    conn.execute(
        "INSERT INTO requests (ts, question, context_chars, prediction, latency_ms, "
        "numeric_hallucination_rate, status, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (time.time(), question, context_chars, prediction, latency_ms,
         numeric_hallucination_rate, status, error),
    )
    conn.commit()
