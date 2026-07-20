# data/scripts/create_api_key.py
# E-CIP v3.0 — Dev API Key Creation
# Blueprint Section 12 — Fix #14
#
# Generates a random API key, bcrypt-hashes it (api/middleware/auth.py's
# hash_api_key), and inserts the hash into PostgreSQL's api_keys table.
# The raw key is printed ONCE — it is not recoverable afterward, since
# only the bcrypt hash is stored.
#
# Usage:
#   python data/scripts/create_api_key.py --name "dev-local"

from __future__ import annotations

import argparse
import os
import secrets

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://ecip:ecip_dev@localhost:5432/ecip"
)


def create_api_key(name: str) -> str:
    import asyncio

    import asyncpg

    from api.middleware.auth import hash_api_key

    raw_key = f"ecip_{secrets.token_urlsafe(32)}"
    key_hash = hash_api_key(raw_key)

    async def _insert() -> None:
        conn = await asyncpg.connect(POSTGRES_DSN)
        try:
            await conn.execute(
                "INSERT INTO api_keys (name, key_hash) VALUES ($1, $2)",
                name,
                key_hash,
            )
        finally:
            await conn.close()

    asyncio.run(_insert())
    return raw_key


def main() -> None:
    parser = argparse.ArgumentParser(description="E-CIP v3.0 — Create a dev API key")
    parser.add_argument("--name", type=str, default="dev-local", help="Label for this key")
    args = parser.parse_args()

    raw_key = create_api_key(args.name)
    print("=" * 60)
    print("  E-CIP v3.0 — API Key Created")
    print("=" * 60)
    print(f"\n  Name: {args.name}")
    print(f"  Key : {raw_key}")
    print("\n  This key is NOT recoverable — only its bcrypt hash is stored.")
    print("  Use it as the X-API-Key header value.")
    print("=" * 60)


if __name__ == "__main__":
    main()
