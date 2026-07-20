# api/routers/review_queue.py
# E-CIP v3.0 — Human Review Queue API
# Blueprint Section 11/13 — Fix #38
#
# GET  /v1/review-queue                — paginated pending items, filter by module/status
# POST /v1/review-queue/{id}/resolve   — mark an item reviewed or dismissed

from __future__ import annotations

import json
import os
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://ecip:ecip_dev@localhost:5432/ecip"
)

router = APIRouter(prefix="/v1/review-queue", tags=["Review Queue"])


class ReviewQueueItem(BaseModel):
    id: str
    request_id: str
    module: str
    trigger: str
    payload: dict[str, Any]
    status: str
    created_at: str
    reviewed_at: str | None


class ReviewQueueResponse(BaseModel):
    items: list[ReviewQueueItem]
    total: int
    depth: int  # count of currently-pending items, regardless of this page's filter


class ResolveRequest(BaseModel):
    resolution: Literal["resolved", "dismissed"] = "resolved"


class ResolveResponse(BaseModel):
    id: str
    status: str


@router.get("", response_model=ReviewQueueResponse)
async def list_review_queue(
    module: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
) -> ReviewQueueResponse:
    import asyncpg

    from observability.prometheus.metrics import set_review_queue_depth

    conn = await asyncpg.connect(POSTGRES_DSN)
    try:
        filters = []
        params: list[Any] = []
        if module:
            params.append(module)
            filters.append(f"module = ${len(params)}")
        if status:
            params.append(status)
            filters.append(f"status = ${len(params)}")
        where = f"WHERE {' AND '.join(filters)}" if filters else ""

        rows = await conn.fetch(
            f"""
            SELECT id, request_id, module, trigger, payload, status, created_at, reviewed_at
            FROM review_queue
            {where}
            ORDER BY created_at DESC
            LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
            """,
            *params,
            limit,
            offset,
        )
        total_row = await conn.fetchrow(
            f"SELECT COUNT(*) AS n FROM review_queue {where}", *params
        )
        depth_row = await conn.fetchrow(
            "SELECT COUNT(*) AS n FROM review_queue WHERE status = 'pending'"
        )
    finally:
        await conn.close()

    depth = int(depth_row["n"]) if depth_row else 0
    set_review_queue_depth(depth)

    items = [
        ReviewQueueItem(
            id=str(row["id"]),
            request_id=row["request_id"],
            module=row["module"],
            trigger=row["trigger"],
            payload=json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"],
            status=row["status"],
            created_at=row["created_at"].isoformat(),
            reviewed_at=row["reviewed_at"].isoformat() if row["reviewed_at"] else None,
        )
        for row in rows
    ]

    return ReviewQueueResponse(
        items=items,
        total=int(total_row["n"]) if total_row else 0,
        depth=depth,
    )


@router.post("/{item_id}/resolve", response_model=ResolveResponse)
async def resolve_review_item(item_id: str, request: ResolveRequest) -> ResolveResponse:
    import asyncpg

    conn = await asyncpg.connect(POSTGRES_DSN)
    try:
        result = await conn.execute(
            """
            UPDATE review_queue
            SET status = $1, reviewed_at = NOW()
            WHERE id = $2::uuid
            """,
            request.resolution,
            item_id,
        )
    finally:
        await conn.close()

    if result == "UPDATE 0":
        raise HTTPException(status_code=404, detail=f"Review queue item {item_id!r} not found")

    return ResolveResponse(id=item_id, status=request.resolution)
