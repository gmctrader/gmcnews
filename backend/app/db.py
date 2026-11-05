"""
Shared Postgres connection for users.py and profiles.py.

Replaces the old local-SQLite-file approach. That worked fine for
local dev, but on Render's free web-service tier the disk is
ephemeral -- every deploy wipes it, which is why accounts and profiles
kept disappearing after each redeploy.

Point DATABASE_URL at any free-tier Postgres (Supabase, Neon, Render's
own free Postgres, ElephantSQL, etc.) and data survives redeploys,
restarts, and scaling events -- it's a separate managed service, not
local disk.

Falls back to a local SQLite file (old behavior) if DATABASE_URL isn't
set, so local dev with zero setup still works exactly as before -- you
just won't want that fallback in production.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
USE_POSTGRES = bool(DATABASE_URL)

_SQLITE_PATH = (
    Path("/tmp/pipx-app.db") if os.environ.get("VERCEL")
    else Path(__file__).parent.parent / "data" / "app.db"
)
_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _pg_conn():
    import psycopg
    # Supabase/Neon connection strings sometimes start with "postgres://";
    # psycopg wants "postgresql://" -- normalize so either works.
    url = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    return psycopg.connect(url, autocommit=True)


@contextmanager
def get_conn():
    """
    Yields a DB connection. Callers write plain SQL using `?` placeholders
    and get an object with .execute()/.fetchone()/.fetchall() either way --
    _Cursor below translates `?` to `%s` for Postgres transparently, so
    users.py/profiles.py don't need two code paths.
    """
    if USE_POSTGRES:
        conn = _pg_conn()
        try:
            yield _PGConnWrapper(conn)
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(_SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()


class _PGConnWrapper:
    """Makes a psycopg connection support the same `?`-placeholder,
    dict-like-row calling convention the rest of the code already uses
    for sqlite3, so users.py/profiles.py stay backend-agnostic."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql: str, params: tuple = ()):
        cur = self._conn.cursor()
        cur.execute(sql.replace("?", "%s"), params)
        return _PGCursorWrapper(cur)


class _PGCursorWrapper:
    def __init__(self, cur):
        self._cur = cur

    def fetchone(self):
        row = self._cur.fetchone()
        return dict(zip(self._columns(), row)) if row else None

    def fetchall(self):
        cols = self._columns()
        return [dict(zip(cols, row)) for row in self._cur.fetchall()]

    def _columns(self):
        return [d[0] for d in self._cur.description] if self._cur.description else []


def is_postgres() -> bool:
    return USE_POSTGRES
