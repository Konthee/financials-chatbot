"""FastAPI application shell: CORS, lifespan (create tables + seed demo user), routers."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from financial_qa.app.api.routes import auth, chat, health
from financial_qa.app.infrastructure import models  # noqa: F401  (register tables on Base.metadata)
from financial_qa.app.infrastructure.db import Base, SessionLocal, engine
from financial_qa.app.infrastructure.models import User
from financial_qa.app.infrastructure.security import hash_password
from financial_qa.app.infrastructure.settings import get_settings


async def _seed_demo_user() -> None:
    settings = get_settings()
    async with SessionLocal() as session:
        existing = (await session.execute(select(User).where(User.email == settings.demo_user_email))).scalar_one_or_none()
        if existing is None:
            session.add(
                User(
                    email=settings.demo_user_email,
                    password_hash=hash_password(settings.demo_user_password),
                )
            )
            await session.commit()


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _seed_demo_user()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Financial QA Chatbot", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(chat.router)
    return app


app = create_app()
