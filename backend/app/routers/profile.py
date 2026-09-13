from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import profiles as db

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


@router.post("")
def create(body: ProfileIn):
    profile = db.create_profile(db.TraderProfile(**body.model_dump()))
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
