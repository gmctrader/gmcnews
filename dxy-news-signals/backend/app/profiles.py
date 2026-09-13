"""
Trader profiles. One trader = one row. SQLite, zero external services.

The onboarding flow (see routers/profile.py) collects just enough to
personalize the feed: experience level, the pairs they actually watch,
risk appetite, and which sessions they trade -- nothing that needs a
broker connection to answer.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "app.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, coltype: str) -> None:
    cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                experience TEXT NOT NULL,
                risk_appetite TEXT NOT NULL,
                pairs TEXT NOT NULL,
                sessions TEXT NOT NULL,
                account_size TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        # Added for account-based auth: links a profile to a signed-up user,
        # and persists the timezone onboarding already collects client-side.
        # Uses ALTER TABLE so this also migrates an existing deployed DB.
        _ensure_column(conn, "profiles", "user_id", "TEXT")
        _ensure_column(conn, "profiles", "timezone", "TEXT")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_profiles_user_id "
            "ON profiles(user_id) WHERE user_id IS NOT NULL"
        )


@dataclass
class TraderProfile:
    name: str
    experience: str          # "new" | "intermediate" | "experienced"
    risk_appetite: str       # "conservative" | "balanced" | "aggressive"
    pairs: list[str] = field(default_factory=list)
    sessions: list[str] = field(default_factory=list)  # "asia" | "london" | "new_york"
    account_size: str = ""
    timezone: str = "UTC"
    user_id: str = ""
    id: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "experience": self.experience,
            "risk_appetite": self.risk_appetite, "pairs": self.pairs,
            "sessions": self.sessions, "account_size": self.account_size,
            "timezone": self.timezone,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }

    @staticmethod
    def _from_row(row: sqlite3.Row) -> "TraderProfile":
        keys = row.keys()
        return TraderProfile(
            id=row["id"], name=row["name"], experience=row["experience"],
            risk_appetite=row["risk_appetite"], pairs=json.loads(row["pairs"]),
            sessions=json.loads(row["sessions"]), account_size=row["account_size"] or "",
            timezone=(row["timezone"] if "timezone" in keys and row["timezone"] else "UTC"),
            user_id=(row["user_id"] if "user_id" in keys and row["user_id"] else ""),
            created_at=row["created_at"], updated_at=row["updated_at"],
        )


def create_profile(p: TraderProfile) -> TraderProfile:
    p.id = p.id or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    p.created_at = p.created_at or now
    p.updated_at = now
    with _conn() as conn:
        conn.execute(
            """INSERT INTO profiles
               (id, name, experience, risk_appetite, pairs, sessions, account_size, timezone, user_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (p.id, p.name, p.experience, p.risk_appetite,
             json.dumps(p.pairs), json.dumps(p.sessions), p.account_size,
             p.timezone, p.user_id or None, p.created_at, p.updated_at),
        )
    return p


def get_profile(profile_id: str) -> TraderProfile | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    return TraderProfile._from_row(row) if row else None


def get_profile_by_user(user_id: str) -> TraderProfile | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
    return TraderProfile._from_row(row) if row else None


def update_profile(profile_id: str, **fields) -> TraderProfile | None:
    existing = get_profile(profile_id)
    if not existing:
        return None
    for k, v in fields.items():
        if hasattr(existing, k) and v is not None:
            setattr(existing, k, v)
    existing.updated_at = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            """UPDATE profiles SET name=?, experience=?, risk_appetite=?,
               pairs=?, sessions=?, account_size=?, timezone=?, updated_at=? WHERE id=?""",
            (existing.name, existing.experience, existing.risk_appetite,
             json.dumps(existing.pairs), json.dumps(existing.sessions),
             existing.account_size, existing.timezone, existing.updated_at, existing.id),
        )
    return existing


def upsert_profile_for_user(
    user_id: str, *, name: str, experience: str, risk_appetite: str,
    pairs: list[str], sessions: list[str], account_size: str = "", timezone_name: str = "UTC",
) -> TraderProfile:
    """Create the user's profile row if this is their first save, else update it in place."""
    existing = get_profile_by_user(user_id)
    if existing:
        return update_profile(
            existing.id, name=name, experience=experience, risk_appetite=risk_appetite,
            pairs=pairs, sessions=sessions, account_size=account_size, timezone=timezone_name,
        )
    return create_profile(TraderProfile(
        name=name, experience=experience, risk_appetite=risk_appetite,
        pairs=pairs, sessions=sessions, account_size=account_size,
        timezone=timezone_name, user_id=user_id,
    ))
