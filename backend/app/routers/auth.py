from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from .. import profiles as profile_db
from .. import users as user_db
from ..security import create_token, decode_token, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])
bearer_scheme = HTTPBearer(auto_error=False)


class RegisterIn(BaseModel):
    email: str
    password: str
    name: str = "Trader"


class LoginIn(BaseModel):
    email: str
    password: str


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> user_db.User:
    """FastAPI dependency: validates the Bearer token, returns the current user or 401s."""
    if creds is None:
        raise HTTPException(401, "Not authenticated")
    user_id = decode_token(creds.credentials)
    if not user_id:
        raise HTTPException(401, "Invalid or expired session")
    user = user_db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(401, "Invalid or expired session")
    return user


def _session_payload(user: user_db.User) -> dict:
    profile = profile_db.get_profile_by_user(user.id)
    return {
        "token": create_token(user.id),
        "user": user.to_public_dict(),
        "profile": profile.to_dict() if profile else None,
    }


@router.post("/register")
def register(body: RegisterIn):
    email = body.email.strip().lower()
    if "@" not in email or "." not in email.split("@")[-1] or len(email) < 5:
        raise HTTPException(400, "Enter a valid email address.")
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")
    if user_db.get_user_by_email(email):
        raise HTTPException(400, "An account with that email already exists.")
    user = user_db.create_user(email, hash_password(body.password), body.name)
    return _session_payload(user)


@router.post("/login")
def login(body: LoginIn):
    user = user_db.get_user_by_email(body.email)
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Incorrect email or password.")
    return _session_payload(user)


@router.get("/me")
def me(current_user: user_db.User = Depends(get_current_user)):
    profile = profile_db.get_profile_by_user(current_user.id)
    return {"user": current_user.to_public_dict(), "profile": profile.to_dict() if profile else None}
