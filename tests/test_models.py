"""Smoke tests for Pydantic models — validates field constraints and serialisation."""

import pytest
from pydantic import ValidationError

from app.models.chat import ChatRequest
from app.models.memory import ContextBundle, MemoryEntry, UserMemoryState


def _memory_entry(**kwargs) -> dict:
    base = {
        "id": "mem_001",
        "subject_id": "user_1",
        "kind": "profile_fact",
        "content": "Senior engineer.",
        "score": 0.98,
        "created_at": "2024-06-01T00:00:00Z",
    }
    base.update(kwargs)
    return base


def test_memory_entry_valid() -> None:
    entry = MemoryEntry(**_memory_entry())
    assert entry.kind == "profile_fact"
    assert entry.score == 0.98
    assert entry.tags == []


def test_memory_entry_score_non_negative() -> None:
    with pytest.raises(ValidationError):
        MemoryEntry(**_memory_entry(score=-0.1))


def test_context_bundle_defaults() -> None:
    bundle = ContextBundle(subject_id="user_1")
    assert bundle.facts == []
    assert bundle.token_estimate == 0
    assert bundle.assembled_context == ""
    assert bundle.memories == []  # alias property


def test_context_bundle_memories_alias() -> None:
    entry = MemoryEntry(**_memory_entry())
    bundle = ContextBundle(subject_id="user_1", facts=[entry])
    assert bundle.memories == bundle.facts


def test_user_memory_state_by_type() -> None:
    entries = [
        MemoryEntry(**_memory_entry(id="m1", kind="profile_fact")),
        MemoryEntry(**_memory_entry(id="m2", kind="preference")),
        MemoryEntry(**_memory_entry(id="m3", kind="preference")),
    ]
    state = UserMemoryState(
        user_id="user_1",
        total_memories=3,
        memories_by_type={"profile_fact": 1, "preference": 2},
        entries=entries,
    )
    assert state.memories_by_type["preference"] == 2


def test_chat_request_empty_message_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(user_id="user_1", message="")


def test_chat_request_valid() -> None:
    req = ChatRequest(user_id="user_1", message="What is my plan tier?")
    assert req.user_id == "user_1"
