"""Pydantic models representing Statewave memory structures."""

from typing import Any

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """A single compiled memory fact returned by Statewave."""

    id: str
    subject_id: str
    kind: str  # profile_fact, preference, open_issue, episode_summary, procedure, etc.
    content: str
    score: float = Field(ge=0.0)
    source_episode_id: str | None = None
    created_at: str
    tags: list[str] = Field(default_factory=list)


class ContextBundle(BaseModel):
    """The assembled context Statewave returns for a given subject and token budget."""

    subject_id: str
    facts: list[MemoryEntry] = Field(default_factory=list)
    token_estimate: int = 0
    assembled_context: str = ""
    receipt_id: str | None = None

    # Convenience alias so call sites can iterate bundle.memories
    @property
    def memories(self) -> list[MemoryEntry]:
        return self.facts


class Episode(BaseModel):
    """A recorded conversation turn stored as an episode in Statewave."""

    id: str
    subject_id: str
    source: str
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class CompileResult(BaseModel):
    """Result of POST /v1/memories/compile."""

    subject_id: str
    memories_created: int
    memories: list[MemoryEntry] = Field(default_factory=list)


class UserMemoryState(BaseModel):
    """Full memory state for a subject — returned by GET /api/v1/memory/{user_id}."""

    user_id: str
    total_memories: int
    memories_by_type: dict[str, int]
    entries: list[MemoryEntry]
