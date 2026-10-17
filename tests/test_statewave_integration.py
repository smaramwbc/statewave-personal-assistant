"""Regression tests for the standalone integration helper."""

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import statewave as sw
from starlette.responses import JSONResponse

import statewave_integration


@pytest.fixture()
def mock_sdk(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    client = AsyncMock(spec=sw.AsyncStatewaveClient)
    monkeypatch.setattr(statewave_integration, "_statewave", client)
    return client


@pytest.mark.asyncio
async def test_get_context_maps_assembled_context(mock_sdk: AsyncMock) -> None:
    mock_sdk.get_context.return_value = sw.ContextBundle(
        subject_id="user_1",
        task="hi",
        assembled_context="User likes Python.",
        token_estimate=10,
    )
    context = await statewave_integration._get_context("user_1", "hi", max_tokens=500)

    assert context == "User likes Python."
    mock_sdk.get_context.assert_awaited_once_with(
        "user_1", "hi", max_tokens=500, caller_id="chat", caller_type="chat"
    )


@pytest.mark.asyncio
async def test_get_context_returns_empty_string_on_error(mock_sdk: AsyncMock) -> None:
    mock_sdk.get_context.side_effect = sw.StatewaveAPIError(503, "unavailable", "Service Unavailable")

    context = await statewave_integration._get_context("user_1", "hi")

    assert context == ""


@pytest.mark.asyncio
async def test_record_episode_calls_create_episode(mock_sdk: AsyncMock) -> None:
    mock_sdk.create_episode.return_value = sw.Episode(
        id=uuid.uuid4(),
        subject_id="user_1",
        source="chat",
        type="conversation",
        payload={},
        created_at=datetime(2024, 6, 1, tzinfo=UTC),
    )

    await statewave_integration._record_episode("user_1", "Hello", "Hi!")

    mock_sdk.create_episode.assert_awaited_once()
    _, kwargs = mock_sdk.create_episode.call_args
    assert kwargs["subject_id"] == "user_1"
    assert kwargs["source"] == "chat"
    assert kwargs["type"] == "conversation"
    assert kwargs["payload"]["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]


@pytest.mark.asyncio
async def test_record_episode_is_nonfatal_on_error(mock_sdk: AsyncMock) -> None:
    mock_sdk.create_episode.side_effect = sw.StatewaveConnectionError("connection refused")

    await statewave_integration._record_episode("user_1", "Hello", "Hi!")  # must not raise


@pytest.mark.asyncio
async def test_middleware_returns_original_response_when_body_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[tuple[str, str, str]] = []

    async def fake_get_context(user_id: str, message: str) -> str:
        return "remembered context"

    async def fake_record_episode(user_id: str, message: str, reply: str) -> None:
        recorded.append((user_id, message, reply))

    async def body() -> bytes:
        return b'{"user_id":"user_1","message":"Hello"}'

    monkeypatch.setattr(statewave_integration, "_get_context", fake_get_context)
    monkeypatch.setattr(statewave_integration, "_record_episode", fake_record_episode)

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/chat"),
        state=SimpleNamespace(),
        body=body,
    )
    response = JSONResponse({"response": "Hi!"})
    response.set_cookie("session", "abc")

    async def call_next(_: Any) -> JSONResponse:
        return response

    middleware = statewave_integration.StatewaveMemoryMiddleware(app=lambda *_: None)

    returned = await middleware.dispatch(request, call_next)

    assert returned is response
    assert returned.headers["set-cookie"].startswith("session=abc")
    assert recorded == [("user_1", "Hello", "Hi!")]


@pytest.mark.asyncio
async def test_middleware_skips_recording_when_memory_message_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[tuple[str, str, str]] = []

    async def fake_record_episode(user_id: str, message: str, reply: str) -> None:
        recorded.append((user_id, message, reply))

    async def body() -> bytes:
        return b"{}"

    monkeypatch.setattr(statewave_integration, "_record_episode", fake_record_episode)

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/chat"),
        state=SimpleNamespace(memory_user_id="user_1"),
        body=body,
    )
    response = JSONResponse({"response": "Hi!"})

    async def call_next(_: Any) -> JSONResponse:
        return response

    middleware = statewave_integration.StatewaveMemoryMiddleware(app=lambda *_: None)

    returned = await middleware.dispatch(request, call_next)

    assert returned is response
    assert recorded == []
