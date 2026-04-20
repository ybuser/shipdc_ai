"""Minimal SQLite event store for prototype runtime logs."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS channels (
                    channel_id TEXT PRIMARY KEY,
                    zone_type TEXT NOT NULL DEFAULT '',
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id TEXT NOT NULL,
                    event_time TEXT NOT NULL,
                    state TEXT NOT NULL,
                    confidence REAL,
                    label TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_time TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    channel_id TEXT,
                    notes TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
                );
                """
            )

    def add_channel(self, channel_id: str, zone_type: str = "", description: str = "") -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO channels (channel_id, zone_type, description, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (channel_id, zone_type, description, utc_now_iso()),
            )

    def add_event(
        self,
        channel_id: str,
        event_time: datetime,
        state: str,
        confidence: float | None = None,
        label: str = "",
        notes: str = "",
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO events (channel_id, event_time, state, confidence, label, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (channel_id, event_time.isoformat(), state, confidence, label, notes),
            )

    def add_metric(
        self,
        name: str,
        value: float,
        metric_time: datetime | None = None,
        channel_id: str | None = None,
        notes: str = "",
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO metrics (metric_time, name, value, channel_id, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                ((metric_time or datetime.now(timezone.utc)).isoformat(), name, value, channel_id, notes),
            )
