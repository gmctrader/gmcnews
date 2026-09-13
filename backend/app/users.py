"""
Trader accounts: email + password.

Storage: Postgres if DATABASE_URL is set (recommended for production --
see db.py), otherwise falls back to a local SQLite file for zero-setup
local dev. Same public API as before (create_user, get_user_by_email,
get_user_by_id, init_db, User), so routers/auth.py needs no changes.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from . import db


def init_db() -> None:
    with db.get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


@dataclass
class User:
    id: str
    email: str
    password_hash: str
    name: str
    created_at: str

    def to_public_dict(self) -> dict:
        # Never include password_hash in anything that reaches the frontend.
        return {"id": self.id, "email": self.email, "name": self.name, "created_at": self.created_at}


def create_user(email: str, password_hash: str, name: str) -> User:
    user = User(
        id=str(uuid.uuid4()),
        email=email.strip().lower(),
        password_hash=password_hash,
        name=name.strip() or "Trader",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO users (id, email, password_hash, name, created_at) VALUES (?, ?, ?, ?, ?)",
            (user.id, user.email, user.password_hash, user.name, user.created_at),
        )
    return user


def get_user_by_email(email: str) -> User | None:
    with db.get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    return User(**dict(row)) if row else None


def get_user_by_id(user_id: str) -> User | None:
    with db.get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return User(**dict(row)) if row else None
