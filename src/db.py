"""Database access layer.

Owns the single SQLAlchemy engine used by every other module. Connects to
Neon serverless PostgreSQL, which auto-suspends on inactivity — ``pool_pre_ping``
transparently reconnects stale connections.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCHEMA_PATH = _PROJECT_ROOT / "sql" / "schema.sql"

_engine: Engine | None = None


def get_engine() -> Engine:
    """Return a process-wide SQLAlchemy engine, creating it on first use."""
    global _engine
    if _engine is None:
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "DATABASE_URL is not set. Copy .env.example to .env and fill it in."
            )
        _engine = create_engine(
            url,
            pool_pre_ping=True,   # reconnect after Neon auto-suspend
            pool_recycle=300,     # recycle connections older than 5 min
            future=True,
        )
    return _engine


def init_db() -> None:
    """Create the schema by executing sql/schema.sql."""
    schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    with get_engine().begin() as conn:
        conn.execute(text(schema_sql))
    print(f"Schema applied from {_SCHEMA_PATH}")


# Keep storage inside the Neon free-tier 0.5 GB cap by dropping old articles.
RETENTION_MONTHS = 18


def purge_old_articles(months: int = RETENTION_MONTHS) -> int:
    """Delete articles older than ``months`` and reclaim the freed space.

    Returns the number of rows deleted. The DELETE runs in a transaction;
    the follow-up VACUUM runs in autocommit mode because VACUUM cannot
    execute inside a transaction block. Reclaiming eagerly matters on Neon,
    where dead tuples count toward the storage quota until vacuumed.
    """
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text(
                "DELETE FROM articles "
                "WHERE published_at < now() - make_interval(months => :months)"
            ),
            {"months": months},
        )
        deleted = result.rowcount
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("VACUUM (ANALYZE) articles"))
    return deleted


if __name__ == "__main__":
    init_db()
