from __future__ import annotations

import json
from datetime import datetime, timezone

from .. import db


def init_db() -> None:
    with db.get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_news_history (
                user_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, event_id)
            )
        """)


def put(user_id: str, event_id: str, payload: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with db.get_conn() as conn:
        conn.execute("""
            INSERT INTO user_news_history (user_id, event_id, payload, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, event_id) DO UPDATE SET
              payload = excluded.payload,
              updated_at = excluded.updated_at
        """, (user_id, event_id, json.dumps(payload), now))


def list_for_user(user_id: str) -> list[dict]:
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT payload FROM user_news_history WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    result = []
    for row in rows:
        try:
            result.append(json.loads(row["payload"]))
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
    return result
