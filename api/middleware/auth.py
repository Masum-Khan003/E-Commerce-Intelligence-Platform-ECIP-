# api/middleware/auth.py
# E-CIP v3.0 — bcrypt API Key Authentication
# Blueprint Section 12 — Fix #14
#
# Keys are bcrypt-hashed in PostgreSQL (db/schema.sql: api_keys.key_hash),
# never stored or logged in plain text. A 5-minute Redis cache avoids
# paying bcrypt's deliberately-slow hash comparison on every request —
# bcrypt is checked once per key per cache window, not once per request.

from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

REDIS_URL = os.environ.get("REDIS_URL_AUTH", "redis://localhost:6379/1")
POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://ecip:ecip_dev@localhost:5432/ecip"
)
CACHE_TTL_SECONDS = 300  # Fix #14: 5-minute Redis cache

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_redis_client: Any = None
_pg_pool: Any = None


def _get_redis() -> Any:
    global _redis_client
    if _redis_client is None:
        import redis

        _redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


async def _get_pg_pool() -> Any:
    global _pg_pool
    if _pg_pool is None:
        import asyncpg

        _pg_pool = await asyncpg.create_pool(POSTGRES_DSN, min_size=1, max_size=5)
    return _pg_pool


async def verify_api_key(
    api_key: str | None = Depends(api_key_header),
) -> str:
    """
    FastAPI dependency: validates X-API-Key against bcrypt hashes stored in
    PostgreSQL's api_keys table, cached in Redis for CACHE_TTL_SECONDS.
    Returns the matching key's id (str) on success; raises 401 otherwise.

    The Redis cache is keyed by SHA-256(api_key), not the raw key — the
    plaintext API key is never written to Redis. A cache hit on that digest
    is sufficient proof of possession (SHA-256 is a one-way function), so
    a cache hit skips bcrypt entirely rather than storing anything that
    could re-derive the original key.
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    import bcrypt

    redis_client = _get_redis()
    cache_digest = _sha256_hex(api_key)
    cache_key = f"apikey:{cache_digest}"

    cached_key_id = redis_client.get(cache_key)
    if cached_key_id:
        return str(cached_key_id)

    pool = await _get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, key_hash FROM api_keys WHERE active = true"
        )

    for row in rows:
        if bcrypt.checkpw(api_key.encode(), row["key_hash"].encode()):
            key_id = str(row["id"])
            redis_client.setex(cache_key, CACHE_TTL_SECONDS, key_id)
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE api_keys SET last_used = NOW() WHERE id = $1", row["id"]
                )
            return key_id

    raise HTTPException(status_code=401, detail="Invalid API key")


def _sha256_hex(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def hash_api_key(raw_key: str) -> str:
    """Bcrypt-hash a raw API key for storage — used by the key-creation script."""
    import bcrypt

    return bcrypt.hashpw(raw_key.encode(), bcrypt.gensalt()).decode()
