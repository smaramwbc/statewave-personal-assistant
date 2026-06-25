"""Integration tests for API routes using FastAPI dependency overrides."""

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.api.routes import get_llm, get_statewave
from app.main import app
from app.models.memory import ContextBundle, Episode, MemoryEntry, UserMemoryState
from app.services.statewave import StatewaveError

client = TestClient(app)


# ── helpers ───────────────────────────────────────────────────────────────────


def _mock_entry(id: str = "mem_001") -> MemoryEntry:
    return MemoryEntry(
        id=id,
        subject_id="user_1",
        kind="profile_fact",
        content="Senior engineer.",
        confidence=0.98,
        source_episode_ids=["ep_001"],
        created_at="2024-06-01T10:00:00Z",
    )


def _mock_bundle(assembled_context: str = "User is a senior engineer.", n_facts: int = 1) -> ContextBundle:
    return ContextBundle(
        subject_id="user_1",
        facts=[_mock_entry(f"mem_{i:03d}") for i in range(n_facts)],
        token_estimate=42,
        assembled_context=assembled_context,
    )


def _mock_episode(episode_id: str = "ep_new") -> Episode:
    return Episode(
        id=episode_id,
        subject_id="user_1",
        source="chat",
        type="conversation",
        payload={},
        created_at="2024-06-01T10:00:00Z",
    )


def _mock_memory_state() -> UserMemoryState:
    return UserMemoryState(
        user_id="user_1",
        total_memories=2,
        memories_by_type={"profile_fact": 1, "episode_summary": 1},
        entries=[
            _mock_entry("mem_001"),
            MemoryEntry(
                id="mem_002",
                subject_id="user_1",
                kind="episode_summary",
                content="Prefers Python.",
                confidence=0.95,
                source_episode_ids=["ep_001"],
                created_at="2024-06-01T10:00:00Z",
            ),
        ],
    )


# ── /health ───────────────────────────────────────────────────────────────────


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── POST /api/v1/chat ─────────────────────────────────────────────────────────


def test_chat_returns_response() -> None:
    sw = AsyncMock()
    sw.get_context = AsyncMock(return_value=_mock_bundle())
    sw.record_episode = AsyncMock(return_value=_mock_episode("ep_abc"))

    llm = AsyncMock()
    llm.chat = AsyncMock(return_value="Here is your answer.")

    app.dependency_overrides[get_statewave] = lambda: sw
    app.dependency_overrides[get_llm] = lambda: llm
    try:
        response = client.post("/api/v1/chat", json={"user_id": "user_1", "message": "Hello"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "user_1"
    assert body["response"] == "Here is your answer."
    assert body["episode_id"] == "ep_abc"
    assert body["context_used"] is True
    assert body["token_estimate"] == 42


def test_chat_context_used_false_when_empty() -> None:
    sw = AsyncMock()
    sw.get_context = AsyncMock(return_value=_mock_bundle(assembled_context=""))
    sw.record_episode = AsyncMock(return_value=_mock_episode())

    llm = AsyncMock()
    llm.chat = AsyncMock(return_value="Generic reply.")

    app.dependency_overrides[get_statewave] = lambda: sw
    app.dependency_overrides[get_llm] = lambda: llm
    try:
        response = client.post("/api/v1/chat", json={"user_id": "user_new", "message": "Hi"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["context_used"] is False


def test_chat_episode_failure_is_nonfatal() -> None:
    sw = AsyncMock()
    sw.get_context = AsyncMock(return_value=_mock_bundle())
    sw.record_episode = AsyncMock(side_effect=StatewaveError(500, "storage error"))

    llm = AsyncMock()
    llm.chat = AsyncMock(return_value="Still answered.")

    app.dependency_overrides[get_statewave] = lambda: sw
    app.dependency_overrides[get_llm] = lambda: llm
    try:
        response = client.post("/api/v1/chat", json={"user_id": "user_1", "message": "Help"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "Still answered."
    assert body["episode_id"] == "unrecorded"


def test_chat_rejects_empty_message() -> None:
    response = client.post("/api/v1/chat", json={"user_id": "user_1", "message": ""})
    assert response.status_code == 422


def test_chat_statewave_down_returns_502() -> None:
    sw = AsyncMock()
    sw.get_context = AsyncMock(side_effect=StatewaveError(503, "service unavailable"))

    app.dependency_overrides[get_statewave] = lambda: sw
    try:
        response = client.post("/api/v1/chat", json={"user_id": "user_1", "message": "Hi"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502


# ── GET /api/v1/memory/{user_id} ─────────────────────────────────────────────


def test_get_memory_returns_state() -> None:
    sw = AsyncMock()
    sw.list_memories = AsyncMock(return_value=_mock_memory_state())

    app.dependency_overrides[get_statewave] = lambda: sw
    try:
        response = client.get("/api/v1/memory/user_1")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "user_1"
    assert body["total_memories"] == 2
    assert len(body["entries"]) == 2
    assert body["memories_by_type"]["profile_fact"] == 1


def test_get_memory_statewave_error_returns_502() -> None:
    sw = AsyncMock()
    sw.list_memories = AsyncMock(side_effect=StatewaveError(404, "subject not found"))

    app.dependency_overrides[get_statewave] = lambda: sw
    try:
        response = client.get("/api/v1/memory/unknown_user")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
