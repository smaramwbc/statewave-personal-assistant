"""Statewave API client.

Thin wrapper around the official `statewave` SDK (`AsyncStatewaveClient`), translating
SDK models into this app's own `app.models.memory` types so route handlers and scripts
don't need to know about the SDK's richer shapes.

Statewave is self-hosted (default: http://localhost:8100).
The API key header is optional; omitted when no key is configured.

Retry policy is delegated to the SDK's `RetryConfig`: transient errors (429, 500, 502, 503,
504, network timeouts) are retried with exponential backoff and jitter.
"""

import logging
from typing import Any

import statewave as sw

from app.core.config import settings
from app.models.memory import CompileResult, ContextBundle, Episode, MemoryEntry, UserMemoryState

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0
_RETRY = sw.RetryConfig(max_retries=3, backoff_base=0.5, jitter=True)
# The server's GET /v1/memories/search endpoint caps limit at 500.
_SEARCH_LIMIT = 500


class StatewaveError(Exception):
    """Raised when the Statewave API returns an unrecoverable error."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        super().__init__(f"Statewave {status_code}: {detail}")


def _wrap(exc: sw.StatewaveError) -> StatewaveError:
    status_code = getattr(exc, "status_code", 0)
    return StatewaveError(status_code, str(exc))


class StatewaveClient:
    """Async wrapper around the official Statewave SDK client."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        resolved_key = api_key if api_key is not None else settings.statewave_api_key
        self._sdk = sw.AsyncStatewaveClient(
            base_url or settings.statewave_base_url,
            _TIMEOUT,
            api_key=resolved_key or None,
            retry=_RETRY,
        )

    async def aclose(self) -> None:
        await self._sdk.close()

    # ── public API ────────────────────────────────────────────────────────────

    async def record_episode(
        self,
        subject_id: str,
        user_message: str,
        assistant_response: str,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        """Record a conversation turn so Statewave can extract and index memories."""
        try:
            episode = await self._sdk.create_episode(
                subject_id=subject_id,
                source="chat",
                type="conversation",
                payload={
                    "messages": [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": assistant_response},
                    ]
                },
                metadata=metadata or {},
            )
        except sw.StatewaveError as exc:
            raise _wrap(exc) from exc
        return Episode(
            id=str(episode.id),
            subject_id=episode.subject_id,
            source=episode.source,
            type=episode.type,
            payload=episode.payload,
            metadata=episode.metadata,
            created_at=episode.created_at.isoformat(),
        )

    async def compile_memories(self, subject_id: str) -> CompileResult:
        """Trigger memory compilation for a subject, waiting until it fully completes."""
        try:
            job = await self._sdk.compile_memories_wait(subject_id, timeout=60.0)
        except sw.StatewaveError as exc:
            raise _wrap(exc) from exc
        return CompileResult(
            subject_id=job.subject_id,
            memories_created=job.memories_created,
            memories=[_to_memory_entry(m) for m in job.memories],
        )

    async def get_context(
        self,
        subject_id: str,
        task: str = "",
        max_tokens: int | None = None,
    ) -> ContextBundle:
        """Retrieve a ranked, token-bounded context bundle for *subject_id*."""
        try:
            bundle = await self._sdk.get_context(
                subject_id,
                task,
                max_tokens=max_tokens or settings.statewave_max_tokens,
            )
        except sw.StatewaveError as exc:
            raise _wrap(exc) from exc
        return ContextBundle(
            subject_id=bundle.subject_id,
            facts=[_to_memory_entry(m) for m in bundle.facts],
            token_estimate=bundle.token_estimate,
            assembled_context=bundle.assembled_context,
            receipt_id=bundle.receipt_id,
        )

    async def list_memories(self, subject_id: str) -> UserMemoryState:
        """Return up to 500 compiled memories for *subject_id*."""
        try:
            result = await self._sdk.search_memories(subject_id, limit=_SEARCH_LIMIT)
        except sw.StatewaveError as exc:
            raise _wrap(exc) from exc
        entries = [_to_memory_entry(m) for m in result.memories]
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


def _to_memory_entry(memory: sw.Memory) -> MemoryEntry:
    return MemoryEntry(
        id=str(memory.id),
        subject_id=memory.subject_id,
        kind=memory.kind,
        content=memory.content,
        confidence=memory.confidence,
        source_episode_ids=[str(i) for i in memory.source_episode_ids],
        created_at=memory.created_at.isoformat(),
        tags=[],
    )
