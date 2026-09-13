from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import profiles as db
from .. import users as user_db
from .auth import get_current_user

router = APIRouter(prefix="/api/profile", tags=["profile"])


class ProfileIn(BaseModel):
    name: str
    experience: str
    risk_appetite: str
    pairs: list[str]
    sessions: list[str]
    account_size: Optional[str] = ""


class ProfilePatch(BaseModel):
    name: Optional[str] = None
    experience: Optional[str] = None
    risk_appetite: Optional[str] = None
    pairs: Optional[list[str]] = None
    sessions: Optional[list[str]] = None
    account_size: Optional[str] = None


class ProfileMeIn(BaseModel):
    """Shape the frontend's onboarding/toggleFavorite flow actually sends to PUT /api/profile/me."""
    name: str
    timezone: Optional[str] = "UTC"
    experience: str
    risk_appetite: str
    pairs: list[str] = []
    sessions: list[str] = []
    account_size: Optional[str] = ""


@router.post("")
def create(body: ProfileIn):
    profile = db.create_profile(db.TraderProfile(**body.model_dump()))
    return profile.to_dict()


# NOTE: these two /me routes must stay registered before GET/PATCH "/{profile_id}"
# below -- otherwise "/api/profile/me" would get captured as profile_id="me".
@router.get("/me")
def read_me(current_user: user_db.User = Depends(get_current_user)):
    profile = db.get_profile_by_user(current_user.id)
    if not profile:
        raise HTTPException(404, "No profile yet -- finish onboarding first.")
    return profile.to_dict()


@router.put("/me")
def upsert_me(body: ProfileMeIn, current_user: user_db.User = Depends(get_current_user)):
    profile = db.upsert_profile_for_user(
        current_user.id,
        name=body.name, experience=body.experience, risk_appetite=body.risk_appetite,
        pairs=body.pairs, sessions=body.sessions,
        account_size=body.account_size or "", timezone_name=body.timezone or "UTC",
    )
    return profile.to_dict()


@router.get("/{profile_id}")
def read(profile_id: str):
    profile = db.get_profile(profile_id)
    if not profile:
        raise HTTPException(404, "Profile not found")
    return profile.to_dict()


@router.patch("/{profile_id}")
def patch(profile_id: str, body: ProfilePatch):
    profile = db.update_profile(profile_id, **body.model_dump(exclude_none=True))
    if not profile:
        raise HTTPException(404, "Profile not found")
    return profile.to_dict()
