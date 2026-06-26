"""Password hashing (bcrypt) and JWT issue/verify."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from passlib.context import CryptContext

from financial_qa.app.infrastructure.settings import get_settings

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# bcrypt rejects inputs longer than 72 bytes; truncate defensively so long passwords still verify.
_BCRYPT_MAX_BYTES = 72


def _clip(password: str) -> str:
    encoded = password.encode("utf-8")
    if len(encoded) <= _BCRYPT_MAX_BYTES:
        return password
    return encoded[:_BCRYPT_MAX_BYTES].decode("utf-8", errors="ignore")


def hash_password(password: str) -> str:
    return _pwd_context.hash(_clip(password))


def verify_password(password: str, password_hash: str) -> bool:
    return _pwd_context.verify(_clip(password), password_hash)


def create_access_token(subject: str) -> str:
    settings = get_settings()
    expire = datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    settings = get_settings()
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
