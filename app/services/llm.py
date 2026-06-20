"""Thin wrapper around the OpenAI chat completions API."""

import logging

import tiktoken
from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

_SYSTEM_BASE = """\
You are a sharp AI coding assistant helping a developer build software.
Answer technical questions directly. Prefer code over prose. Be concise."""

_SYSTEM_WITH_MEMORY = """\
You are a sharp AI coding assistant helping a developer build software.
Answer technical questions directly. Prefer code over prose. Be concise.

## What you already know about this developer
{assembled_context}

Use this context to give a precise, personalised response. \
Do not ask for information you already have."""

# Leave _MAX_REPLY_TOKENS of headroom for the model's reply.
# gpt-4o-mini has a 128k context window.
_MAX_REPLY_TOKENS = 512
_CONTEXT_WINDOW = 128_000
_PROMPT_TOKEN_LIMIT = _CONTEXT_WINDOW - _MAX_REPLY_TOKENS


def _build_system_prompt(assembled_context: str) -> str:
    if assembled_context.strip():
        return _SYSTEM_WITH_MEMORY.format(assembled_context=assembled_context.strip())
    return _SYSTEM_BASE


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


class LLMService:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.llm_api_key)
        self._model = settings.openai_model

    async def chat(self, user_message: str, assembled_context: str = "") -> str:
        """Call the LLM and return the assistant reply as a string.

        Counts tokens before sending. If the prompt would exceed the model's context
        window, assembled_context is stripped and the base prompt is used instead.
        """
        system = _build_system_prompt(assembled_context)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

        token_count = _count_tokens(self._model, messages)

        if token_count > _PROMPT_TOKEN_LIMIT:
            logger.warning(
                "Prompt too large (%d tokens > %d limit); stripping memory context",
                token_count,
                _PROMPT_TOKEN_LIMIT,
            )
            system = _SYSTEM_BASE
            messages[0] = {"role": "system", "content": system}
            token_count = _count_tokens(self._model, messages)

        logger.info(
            "LLM call model=%s prompt_tokens=%d context_chars=%d",
            self._model,
            token_count,
            len(assembled_context),
        )

        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.3,
            max_tokens=_MAX_REPLY_TOKENS,
        )
        content = completion.choices[0].message.content or ""
        return content
