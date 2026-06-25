"""Regression tests for the standalone integration helper."""

from types import SimpleNamespace
from typing import Any

import pytest
from starlette.responses import JSONResponse

import statewave_integration


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
