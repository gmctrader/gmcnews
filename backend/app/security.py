"""
Password hashing and session tokens for the auth routes.

Hashing: PBKDF2-HMAC-SHA256 via the standard library only -- no bcrypt/argon2
C-extension to compile, which keeps this reliable on constrained hosts like
Render's free tier. Stored as "iterations$salt_hex$hash_hex".

Tokens: JWT (PyJWT), signed with AUTH_SECRET_KEY.

IMPORTANT: set AUTH_SECRET_KEY as a real environment variable in Render
(Dashboard -> your service -> Environment). If it's left unset this falls
back to a fixed, publicly-known string -- logins will still work and won't
break on every restart, but anyone who reads this file could forge a valid
session token. Generate a real one with:
    python3 -c "import secrets; print(secrets.token_hex(32))"
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from dotenv import load_dotenv

load_dotenv()

PBKDF2_ITERATIONS = 260_000
TOKEN_TTL_DAYS = 30
_INSECURE_DEFAULT = "dev-insecure-secret-change-me-in-render-env-vars"

_configured_secret = (os.environ.get("AUTH_SECRET_KEY") or "").strip()
AUTH_SECRET_CONFIGURED = len(_configured_secret) >= 32 and _configured_secret != _INSECURE_DEFAULT
SECRET_KEY = _configured_secret if AUTH_SECRET_CONFIGURED else _INSECURE_DEFAULT
if SECRET_KEY == _INSECURE_DEFAULT:
    print(
        "WARNING: AUTH_SECRET_KEY is not set. Using an insecure built-in default -- "
        "set a real random value in your environment before you have real users."
    )


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS)
    return f"{PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        iterations_str, salt, hash_hex = stored.split("$")
        iterations = int(iterations_str)
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations)
    return hmac.compare_digest(digest.hex(), hash_hex)


def create_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": user_id, "iat": now, "exp": now + timedelta(days=TOKEN_TTL_DAYS)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def decode_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    return payload.get("sub")
