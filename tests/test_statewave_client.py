"""Unit tests for the Statewave API client."""

import pytest
from pytest_httpx import HTTPXMock

from app.services.statewave import StatewaveClient, StatewaveError

BASE = "http://localhost:8100"

CONTEXT_PAYLOAD = {
    "subject_id": "user_1",
    "assembled_context": "User is a senior engineer.",
    "token_estimate": 83,
    "facts": [
        {
            "id": "mem_001",
            "subject_id": "user_1",
            "kind": "profile_fact",
            "content": "Senior engineer, enterprise plan.",
            "score": 0.98,
            "source_episode_id": "ep_001",
            "created_at": "2024-06-01T10:00:00Z",
            "tags": ["role"],
        }
    ],
}

EPISODE_PAYLOAD = {
    "id": "ep_xyz",
    "subject_id": "user_1",
    "source": "chat",
    "type": "conversation",
    "payload": {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
    },
    "metadata": {},
    "created_at": "2024-06-01T10:00:00Z",
}

COMPILE_PAYLOAD = {
    "subject_id": "user_1",
    "memories_created": 2,
    "memories": [
        {
            "id": "mem_001",
            "subject_id": "user_1",
            "kind": "profile_fact",
            "content": "Senior engineer.",
            "score": 1.0,
            "source_episode_id": "ep_001",
            "created_at": "2024-06-01T10:00:00Z",
            "tags": [],
        }
    ],
}

MEMORY_SEARCH_PAYLOAD = {
    "memories": [
        {
            "id": "mem_001",
            "subject_id": "user_1",
            "kind": "profile_fact",
            "content": "Senior engineer.",
            "score": 1.0,
            "source_episode_id": "ep_001",
            "created_at": "2024-06-01T10:00:00Z",
            "tags": [],
        }
    ]
}


@pytest.fixture()
def sw() -> StatewaveClient:
    return StatewaveClient(api_key="", base_url=BASE)


@pytest.mark.asyncio
async def test_get_context(httpx_mock: HTTPXMock, sw: StatewaveClient) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/v1/context", json=CONTEXT_PAYLOAD
    )
    bundle = await sw.get_context("user_1", max_tokens=500)

    assert bundle.subject_id == "user_1"
    assert bundle.token_estimate == 83
    assert len(bundle.facts) == 1
    assert bundle.facts[0].kind == "profile_fact"
    assert "senior engineer" in bundle.assembled_context.lower()
    # alias
    assert bundle.memories == bundle.facts


@pytest.mark.asyncio
async def test_record_episode(httpx_mock: HTTPXMock, sw: StatewaveClient) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/v1/episodes", json=EPISODE_PAYLOAD
    )
    episode = await sw.record_episode(
        subject_id="user_1",
        user_message="Hello",
        assistant_response="Hi!",
    )

    assert episode.id == "ep_xyz"
    assert episode.subject_id == "user_1"
    assert episode.source == "chat"


@pytest.mark.asyncio
async def test_compile_memories(httpx_mock: HTTPXMock, sw: StatewaveClient) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/v1/memories/compile", json=COMPILE_PAYLOAD
    )
    result = await sw.compile_memories("user_1")

    assert result.subject_id == "user_1"
    assert result.memories_created == 2
    assert len(result.memories) == 1


@pytest.mark.asyncio
async def test_list_memories(httpx_mock: HTTPXMock, sw: StatewaveClient) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/v1/memories/search?subject_id=user_1",
        json=MEMORY_SEARCH_PAYLOAD,
    )
    state = await sw.list_memories("user_1")

    assert state.total_memories == 1
    assert state.memories_by_type == {"profile_fact": 1}
    assert state.entries[0].id == "mem_001"


@pytest.mark.asyncio
async def test_statewave_error_on_4xx(httpx_mock: HTTPXMock, sw: StatewaveClient) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/v1/context", status_code=401, text="Unauthorized"
    )
    with pytest.raises(StatewaveError) as exc_info:
        await sw.get_context("user_1")

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_statewave_error_on_5xx(httpx_mock: HTTPXMock, sw: StatewaveClient) -> None:
    # 503 is retryable; register one response per attempt (initial + _MAX_RETRIES).
    # The client must exhaust all retries before raising StatewaveError.
    for _ in range(4):  # 1 initial + 3 retries
        httpx_mock.add_response(
            method="POST", url=f"{BASE}/v1/context", status_code=503, text="Service Unavailable"
        )
    with pytest.raises(StatewaveError) as exc_info:
        await sw.get_context("user_1")

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_no_auth_header_when_no_key(httpx_mock: HTTPXMock) -> None:
    """When no API key is set, X-Api-Key header must NOT be sent."""
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/v1/context", json=CONTEXT_PAYLOAD
    )
    sw = StatewaveClient(api_key="", base_url=BASE)
    await sw.get_context("user_1")
    request = httpx_mock.get_requests()[0]
    assert "x-api-key" not in {k.lower() for k in request.headers}
    await sw.aclose()


@pytest.mark.asyncio
async def test_auth_header_sent_when_key_set(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/v1/context", json=CONTEXT_PAYLOAD
    )
    sw = StatewaveClient(api_key="sw-test-key", base_url=BASE)
    await sw.get_context("user_1")
    request = httpx_mock.get_requests()[0]
    assert request.headers.get("x-api-key") == "sw-test-key"
    await sw.aclose()


@pytest.mark.asyncio
async def test_context_manager(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/v1/context", json=CONTEXT_PAYLOAD
    )
    async with StatewaveClient(api_key="", base_url=BASE) as sw:
        bundle = await sw.get_context("user_1")
    assert bundle.token_estimate == 83
