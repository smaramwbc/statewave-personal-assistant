"""
statewave_integration.py
========================
Drop-in Statewave memory layer for any OpenAI-based chatbot.

Copy this file into your project. Three patterns are included — use whichever
matches your stack:

  Pattern 1 — Plain async function wrapper   (works with any framework)
  Pattern 2 — FastAPI middleware              (zero changes to existing routes)
  Pattern 3 — LangChain RunnableLambda       (drop into any LCEL chain)

Requirements:
  pip install statewave openai tiktoken

Environment variables:
  STATEWAVE_BASE_URL   default: http://localhost:8100
  STATEWAVE_API_KEY    optional (leave blank for self-hosted)
  OPENAI_API_KEY
  OPENAI_MODEL         default: gpt-4o-mini
"""

from __future__ import annotations

import json as _json
import logging
import os
from typing import Any

import statewave as sw
import tiktoken
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

STATEWAVE_BASE_URL: str = os.getenv("STATEWAVE_BASE_URL", "http://localhost:8100")
STATEWAVE_API_KEY:  str = os.getenv("STATEWAVE_API_KEY", "")
OPENAI_MODEL:       str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_MEMORY_TOKENS:  int = int(os.getenv("STATEWAVE_MAX_TOKENS", "800"))

# Leave headroom for the model's reply so an oversized prompt+context never
# silently overflows the context window.
_MAX_REPLY_TOKENS = 512
_CONTEXT_WINDOW = 128_000
_PROMPT_TOKEN_LIMIT = _CONTEXT_WINDOW - _MAX_REPLY_TOKENS

_openai: AsyncOpenAI | None = None
_statewave: sw.AsyncStatewaveClient | None = None


def _openai_client() -> AsyncOpenAI:
    global _openai
    if _openai is None:
        _openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    return _openai


def _statewave_client() -> sw.AsyncStatewaveClient:
    global _statewave
    if _statewave is None:
        _statewave = sw.AsyncStatewaveClient(
            STATEWAVE_BASE_URL,
            api_key=STATEWAVE_API_KEY or None,
        )
    return _statewave


def _count_tokens(model: str, messages: list[dict[str, str]]) -> int:
    """Estimate token count for a list of chat messages using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    total = 0
    for msg in messages:
        total += 4  # per-message overhead (role + separators)
        total += len(enc.encode(msg.get("content", "")))
    total += 2  # reply priming tokens
    return total

# ── low-level Statewave helpers ───────────────────────────────────────────────


async def _get_context(user_id: str, message: str, max_tokens: int = MAX_MEMORY_TOKENS) -> str:
    """Return the assembled memory context string for *user_id*, or '' on failure."""
    try:
        bundle = await _statewave_client().get_context(
            user_id,
            message,
            max_tokens=max_tokens,
            caller_id="personal_assistant",
            caller_type="chat",
        )
        return bundle.assembled_context
    except sw.StatewaveError as exc:
        logger.warning("Statewave get_context failed for %s: %s", user_id, exc)
        return ""


async def _record_episode(user_id: str, user_message: str, assistant_response: str) -> None:
    """Record a conversation turn as a Statewave episode. Non-fatal."""
    try:
        await _statewave_client().create_episode(
            subject_id=user_id,
            source="chat",
            type="conversation",
            payload={
                "messages": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_response},
                ]
            },
        )
    except sw.StatewaveError as exc:
        logger.warning("Statewave record_episode failed for %s: %s", user_id, exc)


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 1 — Plain async wrapper
# ══════════════════════════════════════════════════════════════════════════════
#
# Before (your existing code):
#
#   reply = await plain_chat(user_id="alice", message="How do I paginate results?")
#
# After (this wrapper replaces it with zero other changes):
#
#   reply = await memory_chat(user_id="alice", message="How do I paginate results?")
#
# That's it. Memory is fetched before the LLM call, recorded after.

async def memory_chat(
    user_id: str,
    message: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = OPENAI_MODEL,
) -> str:
    """Drop-in replacement for a plain openai.chat call with Statewave memory."""

    # 1. Retrieve ranked, token-bounded memory context.
    context = await _get_context(user_id, message)

    # 2. Build system prompt — inject memory if available.
    if context.strip():
        system = f"{system_prompt}\n\n## What you know about this user\n{context}\n\nUse this context. Do not ask for information you already have."
    else:
        system = system_prompt

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": message},
    ]

    # 2b. Guard against blowing the context window — drop memory context if needed
    # rather than letting the request fail outright.
    if _count_tokens(model, messages) > _PROMPT_TOKEN_LIMIT:
        logger.warning("Prompt too large for %s; stripping memory context", user_id)
        messages[0] = {"role": "system", "content": system_prompt}

    # 3. Call the LLM.
    completion = await _openai_client().chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.3,
        max_tokens=_MAX_REPLY_TOKENS,
    )
    reply: str = completion.choices[0].message.content or ""

    # 4. Record the turn so Statewave can extract new memories.
    await _record_episode(user_id, message, reply)

    return reply


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 2 — FastAPI middleware
# ══════════════════════════════════════════════════════════════════════════════
#
# Intercepts every POST to /chat, injects memory into the request before your
# handler sees it, then records the episode after your handler responds.
#
# 1. Add to your FastAPI app:
#
#       from statewave_integration import StatewaveMemoryMiddleware
#       app.add_middleware(StatewaveMemoryMiddleware)
#
# 2. Your route handler stays exactly as-is:
#
#       @app.post("/chat")
#       async def chat(req: ChatRequest):
#           reply = await openai.chat(messages=[
#               {"role": "system",  "content": req.injected_context},  # ← added by middleware
#               {"role": "user",    "content": req.message},
#           ])
#           return {"response": reply}
#
# The middleware attaches `request.state.memory_context` so you can use it
# inside any route handler without changing the request model.

try:
    from fastapi import Request, Response
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import Response as StarletteResponse
    from starlette.types import ASGIApp

    class StatewaveMemoryMiddleware(BaseHTTPMiddleware):
        """Injects Statewave memory context into POST /chat requests."""

        def __init__(self, app: ASGIApp, chat_path: str = "/chat") -> None:
            super().__init__(app)
            self._chat_path = chat_path

        async def dispatch(self, request: Request, call_next: Any) -> Response:
            if request.method == "POST" and request.url.path.endswith(self._chat_path):
                body_bytes = await request.body()
                try:
                    body: dict[str, Any] = _json.loads(body_bytes)
                    user_id: str  = body.get("user_id", "")
                    message: str  = body.get("message", "")
                    if user_id and message:
                        ctx = await _get_context(user_id, message)
                        request.state.memory_context  = ctx
                        request.state.memory_user_id  = user_id
                        request.state.memory_message  = message
                except Exception:
                    pass

                response = await call_next(request)

                # Record episode after response (best-effort; never swallows the body).
                if hasattr(response, "body"):
                    try:
                        resp_data: dict[str, Any] = _json.loads(response.body)
                        assistant_reply: str = resp_data.get("response", "")
                        user_id = getattr(request.state, "memory_user_id", None)
                        message = getattr(request.state, "memory_message", None)
                        if user_id and message and assistant_reply:
                            await _record_episode(
                                user_id,
                                message,
                                assistant_reply,
                            )
                    except Exception:
                        pass
                    return response

                resp_body = b""
                async for chunk in response.body_iterator:  # type: ignore[attr-defined]
                    resp_body += chunk

                try:
                    resp_data = _json.loads(resp_body)
                    assistant_reply = resp_data.get("response", "")
                    user_id = getattr(request.state, "memory_user_id", None)
                    message = getattr(request.state, "memory_message", None)
                    if user_id and message and assistant_reply:
                        await _record_episode(
                            user_id,
                            message,
                            assistant_reply,
                        )
                except Exception:
                    pass

                return StarletteResponse(
                    content=resp_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )

            return await call_next(request)

except ImportError:
    # FastAPI/Starlette not installed — skip this pattern silently.
    pass


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 3 — LangChain RunnableLambda
# ══════════════════════════════════════════════════════════════════════════════
#
# Drop into any LCEL chain. The lambda fetches memory, injects it into the
# ChatPromptTemplate variables, and records the episode after the chain runs.
#
#   from statewave_integration import statewave_memory_step
#   from langchain_openai import ChatOpenAI
#   from langchain_core.prompts import ChatPromptTemplate
#   from langchain_core.output_parsers import StrOutputParser
#
#   prompt = ChatPromptTemplate.from_messages([
#       ("system", "You are a helpful assistant.\n\n{memory_context}"),
#       ("human", "{message}"),
#   ])
#   chain = statewave_memory_step | prompt | ChatOpenAI() | StrOutputParser()
#
#   reply = await chain.ainvoke({"user_id": "alice", "message": "What was I building?"})

try:
    from langchain_core.runnables import RunnableLambda

    async def _inject_memory(inputs: dict[str, Any]) -> dict[str, Any]:
        user_id = inputs.get("user_id", "")
        message = inputs.get("message", "")
        context = await _get_context(user_id, message) if user_id else ""
        return {**inputs, "memory_context": context}

    statewave_memory_step = RunnableLambda(_inject_memory)

except ImportError:
    # langchain_core not installed — skip this pattern silently.
    statewave_memory_step = None  # type: ignore[assignment]


# ══════════════════════════════════════════════════════════════════════════════
# Quick smoke-test — run directly: python statewave_integration.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import asyncio

    async def _smoke() -> None:
        print("Testing memory_chat with user_id='demo_user'…")
        reply = await memory_chat(
            user_id="demo_user",
            message="What's a good first thing to test with Statewave?",
        )
        print(f"\nReply:\n{reply}")

    asyncio.run(_smoke())
