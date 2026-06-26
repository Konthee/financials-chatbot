"""Current-user profile."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from financial_qa.app.api.deps import get_current_user
from financial_qa.app.infrastructure.models import User

router = APIRouter(prefix="/api/v1/users", tags=["users"])


class UserProfile(BaseModel):
    email: str
    created_at: datetime


@router.get("/me", response_model=UserProfile)
async def me(user: Annotated[User, Depends(get_current_user)]) -> UserProfile:
    return UserProfile(email=user.email, created_at=user.created_at)
