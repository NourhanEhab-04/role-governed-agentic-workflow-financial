# orchestrator/db.py
"""
Async PostgreSQL connection pool and migration runner.

DATABASE_URL must be set as an environment variable:
  postgresql://user:password@host:port/dbname

The pool is created on first use and shared for the process lifetime.
Call close_pool() at application shutdown to drain connections cleanly.
"""

import os
from pathlib import Path

_pool = None
_MIGRATIONS_DIR = Path(__file__).parent.parent / "db" / "migrations"


def _require_asyncpg():
    try:
        import asyncpg
        return asyncpg
    except ImportError:
        raise ImportError(
            "asyncpg is required for PostgreSQL audit persistence. "
            "Install it with: pip install asyncpg"
        )


async def get_pool():
    """Return (or lazily create) the shared asyncpg connection pool."""
    global _pool
    if _pool is None:
        asyncpg = _require_asyncpg()
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "DATABASE_URL environment variable is not set. "
                "Set it to a PostgreSQL connection string to enable audit persistence."
            )
        _pool = await asyncpg.create_pool(url, min_size=1, max_size=10)
    return _pool


async def close_pool():
    """Close the shared pool gracefully. Call at application shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def apply_migrations(pool=None):
    """
    Execute all SQL files in db/migrations/ in lexicographic order.

    Safe to call on every startup — each migration uses CREATE TABLE IF NOT EXISTS
    and CREATE INDEX IF NOT EXISTS so re-running is idempotent.
    """
    if pool is None:
        pool = await get_pool()
    migration_files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
    if not migration_files:
        return
    async with pool.acquire() as conn:
        for path in migration_files:
            sql = path.read_text(encoding="utf-8")
            await conn.execute(sql)
