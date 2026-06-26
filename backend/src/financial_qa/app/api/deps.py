"""Shared FastAPI dependencies: DB session and current authenticated user."""

from __future__ import annotations

from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_qa.app.infrastructure.db import get_session
from financial_qa.app.infrastructure.models import User
from financial_qa.app.infrastructure.security import decode_token

_bearer = HTTPBearer(auto_error=False)

_UNAUTHORIZED = {"WWW-Authenticate": "Bearer"}


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token", headers=_UNAUTHORIZED)
    try:
        email = decode_token(credentials.credentials).get("sub")
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token", headers=_UNAUTHORIZED)
    if not email:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token", headers=_UNAUTHORIZED)

    user = (await session.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found or inactive", headers=_UNAUTHORIZED)
    return user
