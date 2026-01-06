"""Database connection and session management."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session

from app.config import DATABASE_URL, DATABASE_URL_SYNC
from app.database.models import Base


# Async engine for FastAPI
async_engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Sync engine for scripts and ML training
sync_engine = create_engine(
    DATABASE_URL_SYNC,
    echo=False,
    future=True,
)

SyncSessionLocal = sessionmaker(
    sync_engine,
    class_=Session,
    expire_on_commit=False,
)


async def init_db():
    """Initialize database tables."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def init_db_sync():
    """Initialize database tables (synchronous version)."""
    Base.metadata.create_all(bind=sync_engine)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_sync_session() -> Session:
    """Get synchronous database session for scripts."""
    return SyncSessionLocal()
