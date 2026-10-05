"""Unit tests for the Statewave API client wrapper (mocks the statewave SDK)."""

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import statewave as sw

from app.services.statewave import StatewaveClient, StatewaveError

BASE = "http://localhost:8100"

MEM_ID = uuid.uuid4()
EP_ID = uuid.uuid4()
NOW = datetime(2024, 6, 1, 10, 0, 0, tzinfo=UTC)


def make_memory(**overrides: Any) -> sw.Memory:
    defaults: dict[str, Any] = dict(
        id=MEM_ID,
        subject_id="user_1",
        kind="profile_fact",
        content="Senior engineer.",
        confidence=1.0,
        valid_from=NOW,
        source_episode_ids=[EP_ID],
        created_at=NOW,
        updated_at=NOW,
    )
    defaults.update(overrides)
    return sw.Memory(**defaults)


def make_episode(**overrides: Any) -> sw.Episode:
    defaults: dict[str, Any] = dict(
        id=EP_ID,
        subject_id="user_1",
        source="chat",
        type="conversation",
        payload={"messages": [{"role": "user", "content": "Hello"}]},
        created_at=NOW,
    )
    defaults.update(overrides)
    return sw.Episode(**defaults)


@pytest.fixture()
def mock_sdk() -> AsyncMock:
    return AsyncMock(spec=sw.AsyncStatewaveClient)


@pytest.fixture()
def sw_client(mock_sdk: AsyncMock) -> StatewaveClient:
    with patch("app.services.statewave.sw.AsyncStatewaveClient", return_value=mock_sdk):
        client = StatewaveClient(api_key="", base_url=BASE)
    client._sdk = mock_sdk
    return client


@pytest.mark.asyncio
async def test_get_context(sw_client: StatewaveClient, mock_sdk: AsyncMock) -> None:
    mock_sdk.get_context.return_value = sw.ContextBundle(
        subject_id="user_1",
        task="hi",
        facts=[make_memory()],
        assembled_context="User is a senior engineer.",
        token_estimate=83,
        receipt_id="rcpt_1",
    )
    bundle = await sw_client.get_context("user_1", max_tokens=500)

    assert bundle.subject_id == "user_1"
    assert bundle.token_estimate == 83
    assert len(bundle.facts) == 1
    assert bundle.facts[0].kind == "profile_fact"
    assert "senior engineer" in bundle.assembled_context.lower()
    assert bundle.memories == bundle.facts


@pytest.mark.asyncio
async def test_record_episode(sw_client: StatewaveClient, mock_sdk: AsyncMock) -> None:
    mock_sdk.create_episode.return_value = make_episode()
    episode = await sw_client.record_episode(
        subject_id="user_1",
        user_message="Hello",
        assistant_response="Hi!",
    )

    assert episode.id == str(EP_ID)
    assert episode.subject_id == "user_1"
    assert episode.source == "chat"


@pytest.mark.asyncio
async def test_compile_memories(sw_client: StatewaveClient, mock_sdk: AsyncMock) -> None:
    mock_sdk.compile_memories_wait.return_value = sw.CompileJob(
        job_id="job_1",
        status="completed",
        subject_id="user_1",
        memories_created=2,
        memories=[make_memory()],
    )
    result = await sw_client.compile_memories("user_1")

    assert result.subject_id == "user_1"
    assert result.memories_created == 2
    assert len(result.memories) == 1


@pytest.mark.asyncio
async def test_list_memories(sw_client: StatewaveClient, mock_sdk: AsyncMock) -> None:
    mock_sdk.search_memories.return_value = sw.SearchResult(memories=[make_memory()])
    state = await sw_client.list_memories("user_1")

    mock_sdk.search_memories.assert_awaited_once_with("user_1", limit=500)
    assert state.total_memories == 1
    assert state.memories_by_type == {"profile_fact": 1}
    assert state.entries[0].id == str(MEM_ID)


@pytest.mark.asyncio
async def test_list_memories_empty_result(sw_client: StatewaveClient, mock_sdk: AsyncMock) -> None:
    mock_sdk.search_memories.return_value = sw.SearchResult(memories=[])
    state = await sw_client.list_memories("user_1")
    mock_sdk.search_memories.assert_awaited_once_with("user_1", limit=500)
    assert state.total_memories == 0
    assert state.memories_by_type == {}
    assert state.entries == []


@pytest.mark.asyncio
async def test_list_memories_propagates_error(sw_client: StatewaveClient, mock_sdk: AsyncMock) -> None:
    mock_sdk.search_memories.side_effect = sw.StatewaveAPIError(503, "unavailable", "Unavailable")
    with pytest.raises(StatewaveError) as exc_info:
        await sw_client.list_memories("user_1")
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_statewave_error_on_4xx(sw_client: StatewaveClient, mock_sdk: AsyncMock) -> None:
    mock_sdk.get_context.side_effect = sw.StatewaveAPIError(401, "unauthorized", "Unauthorized")
    with pytest.raises(StatewaveError) as exc_info:
        await sw_client.get_context("user_1")

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_statewave_error_on_5xx(sw_client: StatewaveClient, mock_sdk: AsyncMock) -> None:
    mock_sdk.get_context.side_effect = sw.StatewaveAPIError(
        503, "service_unavailable", "Service Unavailable"
    )
    with pytest.raises(StatewaveError) as exc_info:
        await sw_client.get_context("user_1")

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_no_auth_header_when_no_key(mock_sdk: AsyncMock) -> None:
    """When no API key is set, the SDK client must be constructed with api_key=None."""
    with patch(
        "app.services.statewave.sw.AsyncStatewaveClient", return_value=mock_sdk
    ) as ctor:
        StatewaveClient(api_key="", base_url=BASE)
    assert ctor.call_args.kwargs["api_key"] is None


@pytest.mark.asyncio
async def test_auth_header_sent_when_key_set(mock_sdk: AsyncMock) -> None:
    with patch(
        "app.services.statewave.sw.AsyncStatewaveClient", return_value=mock_sdk
    ) as ctor:
        StatewaveClient(api_key="sw-test-key", base_url=BASE)
    assert ctor.call_args.kwargs["api_key"] == "sw-test-key"


@pytest.mark.asyncio
async def test_context_manager(sw_client: StatewaveClient, mock_sdk: AsyncMock) -> None:
    mock_sdk.get_context.return_value = sw.ContextBundle(
        subject_id="user_1",
        task="",
        assembled_context="User is a senior engineer.",
        token_estimate=83,
    )
    async with sw_client as client:
        bundle = await client.get_context("user_1")
    assert bundle.token_estimate == 83
    mock_sdk.close.assert_awaited_once()
