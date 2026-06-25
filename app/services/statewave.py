"""Statewave API client.

Wraps the Statewave calls this demo uses:
  1. POST /v1/episodes          - record a conversation turn as an episode
  2. POST /v1/memories/compile  - compile memories from recorded episodes
  3. POST /v1/context           - retrieve a ranked, token-bounded context bundle
  4. GET  /v1/memories/search   - list/search compiled memories for a subject

Statewave is self-hosted (default: http://localhost:8100).
The API key header (X-Api-Key) is optional; omitted when no key is configured.

Retry policy: transient errors (429, 502, 503, 504, network timeouts) are retried
up to _MAX_RETRIES times with exponential backoff and jitter. 4xx client errors
(except 429) are not retried because they indicate a bad request, not a transient fault.
"""

import asyncio
import logging
import random
from typing import Any

import httpx

from app.core.config import settings
from app.models.memory import CompileResult, ContextBundle, Episode, MemoryEntry, UserMemoryState

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(30.0)
_MAX_RETRIES = 3
_RETRY_BASE_SECONDS = 0.5  # doubles each attempt: 0.5s, 1s, 2s
_RETRYABLE_STATUS = {429, 502, 503, 504}


class StatewaveError(Exception):
    """Raised when the Statewave API returns an unrecoverable error."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        super().__init__(f"Statewave {status_code}: {detail}")


class StatewaveClient:
    """Async HTTP client for the Statewave memory API."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._base_url = (base_url or settings.statewave_base_url).rstrip("/")
        self._api_key = api_key if api_key is not None else settings.statewave_api_key

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-Api-Key"] = self._api_key

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=_TIMEOUT,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── private helpers ───────────────────────────────────────────────────────

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send an HTTP request, retrying on transient failures with exponential backoff."""
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._client.request(method, path, json=json, params=params)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    wait = _RETRY_BASE_SECONDS * (2**attempt) + random.uniform(0, 0.3)
                    logger.warning(
                        "Statewave %s %s network error (attempt %d/%d), retrying in %.2fs: %s",
                        method, path, attempt + 1, _MAX_RETRIES, wait, exc,
                    )
                    await asyncio.sleep(wait)
                continue

            if response.status_code < 400:
                result: dict[str, Any] = response.json()
                return result

            if response.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
                wait = _RETRY_BASE_SECONDS * (2**attempt) + random.uniform(0, 0.3)
                logger.warning(
                    "Statewave %s %s returned %d (attempt %d/%d), retrying in %.2fs",
                    method, path, response.status_code, attempt + 1, _MAX_RETRIES, wait,
                )
                await asyncio.sleep(wait)
                continue

            raise StatewaveError(response.status_code, response.text)

        if last_exc is not None:
            raise StatewaveError(0, f"Network error after {_MAX_RETRIES} retries: {last_exc}")
        raise StatewaveError(0, f"Failed after {_MAX_RETRIES} retries")

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request_with_retry("POST", path, json=payload)

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._request_with_retry("GET", path, params=params)

    # ── public API ────────────────────────────────────────────────────────────

    async def record_episode(
        self,
        subject_id: str,
        user_message: str,
        assistant_response: str,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        """Record a conversation turn so Statewave can extract and index memories.

        Episodes use the standard conversation payload shape:
          source="chat", type="conversation",
          payload={"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
        """
        payload: dict[str, Any] = {
            "subject_id": subject_id,
            "source": "chat",
            "type": "conversation",
            "payload": {
                "messages": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_response},
                ]
            },
            "metadata": metadata or {},
        }
        data = await self._post("/v1/episodes", payload)
        return Episode(
            id=data["id"],
            subject_id=data.get("subject_id", subject_id),
            source=data.get("source", "chat"),
            type=data.get("type", "conversation"),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at"),
        )

    async def compile_memories(self, subject_id: str) -> CompileResult:
        """Trigger memory compilation for a subject.

        Statewave processes uncompiled episodes and extracts structured memory facts.
        Loops until the server's has_more flag is False to handle large episode sets.
        Call this after seeding episodes; in live use the server can auto-compile.
        """
        all_memories: list[MemoryEntry] = []
        total_created = 0
        for _ in range(100):
            data = await self._post("/v1/memories/compile", {"subject_id": subject_id})
            batch = [MemoryEntry(**m) for m in data.get("memories", [])]
            all_memories.extend(batch)
            total_created += data.get("memories_created", len(batch))
            if not data.get("has_more", False):
                break
        else:
            logger.warning("Memory compilation exceeded maximum iteration limit for subject %s", subject_id)
        return CompileResult(
            subject_id=subject_id,
            memories_created=total_created,
            memories=all_memories,
        )

    async def get_context(
        self,
        subject_id: str,
        task: str = "",
        max_tokens: int | None = None,
    ) -> ContextBundle:
        """Retrieve a ranked, token-bounded context bundle for *subject_id*.

        Statewave selects the highest-signal facts that fit within *max_tokens*,
        assembles them into a ready-to-inject string, and returns a token estimate.
        """
        payload: dict[str, Any] = {
            "subject_id": subject_id,
            "max_tokens": max_tokens or settings.statewave_max_tokens,
        }
        if task:
            payload["task"] = task
        data = await self._post("/v1/context", payload)
        facts = [MemoryEntry(**f) for f in data.get("facts", [])]
        return ContextBundle(
            subject_id=data.get("subject_id", subject_id),
            facts=facts,
            token_estimate=data.get("token_estimate", 0),
            assembled_context=data.get("assembled_context", ""),
            receipt_id=data.get("receipt_id"),
        )

    async def list_memories(self, subject_id: str) -> UserMemoryState:
        """Return the full compiled memory state for *subject_id*."""
        data = await self._get("/v1/memories/search", params={"subject_id": subject_id})
        entries = [MemoryEntry(**m) for m in data.get("memories", [])]
        by_type: dict[str, int] = {}
        for entry in entries:
            by_type[entry.kind] = by_type.get(entry.kind, 0) + 1
        return UserMemoryState(
            user_id=subject_id,
            total_memories=len(entries),
            memories_by_type=by_type,
            entries=entries,
        )

    async def __aenter__(self) -> "StatewaveClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
