"""FastAPI route handlers."""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.models.chat import ChatRequest, ChatResponse, CompareResponse, CompareSide
from app.models.memory import UserMemoryState
from app.services.llm import LLMService
from app.services.statewave import StatewaveClient, StatewaveError

logger = logging.getLogger(__name__)

router = APIRouter()


# ── dependency factories ──────────────────────────────────────────────────────


def get_statewave() -> StatewaveClient:
    return StatewaveClient()


def get_llm() -> LLMService:
    return LLMService()


# ── endpoints ─────────────────────────────────────────────────────────────────


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message and get a memory-aware response",
    description=(
        "Retrieves the user's Statewave context bundle, injects it into the LLM system "
        "prompt, returns the personalised response, then records the turn as a new episode "
        "so Statewave can extract and index any new memories."
    ),
)
async def chat(
    req: ChatRequest,
    sw: StatewaveClient = Depends(get_statewave),  # noqa: B008
    llm: LLMService = Depends(get_llm),  # noqa: B008
) -> ChatResponse:
    # 1. Retrieve ranked, token-bounded memory context.
    try:
        bundle = await sw.get_context(req.user_id, task=req.message)
    except StatewaveError as exc:
        logger.error("Statewave context error for %s: %s", req.user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not retrieve memory context: {exc}",
        ) from exc

    logger.info(
        "user=%s context_tokens=%d facts=%d",
        req.user_id,
        bundle.token_estimate,
        len(bundle.facts),
    )

    # 2. Call the LLM with assembled context injected into the system prompt.
    try:
        reply = await llm.chat(req.message, bundle.assembled_context)
    except Exception as exc:
        logger.error("LLM error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM call failed: {exc}",
        ) from exc

    # 3. Record this turn as an episode so Statewave can update its memory index.
    episode_id = "unrecorded"
    try:
        episode = await sw.record_episode(
            subject_id=req.user_id,
            user_message=req.message,
            assistant_response=reply,
        )
        episode_id = episode.id
    except StatewaveError as exc:
        # Non-fatal: reply already generated. Log and continue.
        logger.warning("Failed to record episode for %s: %s", req.user_id, exc)

    return ChatResponse(
        user_id=req.user_id,
        message=req.message,
        response=reply,
        episode_id=episode_id,
        context_used=bool(bundle.assembled_context),
        token_estimate=bundle.token_estimate,
    )


@router.get(
    "/memory/{user_id}",
    response_model=UserMemoryState,
    summary="Inspect compiled memory state for a user",
    description=(
        "Returns all memory facts Statewave has compiled for this subject: "
        "profile facts, procedures, artifact references, and episode summaries — "
        "each with its confidence score, source episode IDs, and timestamp."
    ),
)
async def get_memory(
    user_id: str,
    sw: StatewaveClient = Depends(get_statewave),  # noqa: B008
) -> UserMemoryState:
    try:
        return await sw.list_memories(user_id)
    except StatewaveError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Side-by-side: stateless LLM vs Statewave-powered response",
    description=(
        "Fires both a bare LLM call (no memory) and a Statewave-augmented call in parallel "
        "for the same message, returning both responses so the difference is immediately "
        "visible. The Statewave turn also records an episode as normal."
    ),
)
async def compare(
    req: ChatRequest,
    sw: StatewaveClient = Depends(get_statewave),  # noqa: B008
    llm: LLMService = Depends(get_llm),  # noqa: B008
) -> CompareResponse:
    # Fetch Statewave context (needed for the memory side only).
    try:
        bundle = await sw.get_context(req.user_id, task=req.message)
    except StatewaveError as exc:
        logger.error("Statewave context error for %s: %s", req.user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not retrieve memory context: {exc}",
        ) from exc

    # Fire both LLM calls in parallel — no memory vs with memory.
    try:
        reply_bare, reply_memory = await asyncio.gather(
            llm.chat(req.message, assembled_context=""),
            llm.chat(req.message, assembled_context=bundle.assembled_context),
        )
    except Exception as exc:
        logger.error("LLM error in compare: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM call failed: {exc}",
        ) from exc

    # Record the memory-augmented side as an episode (non-fatal).
    try:
        await sw.record_episode(
            subject_id=req.user_id,
            user_message=req.message,
            assistant_response=reply_memory,
        )
    except StatewaveError as exc:
        logger.warning("Failed to record episode for %s: %s", req.user_id, exc)

    return CompareResponse(
        user_id=req.user_id,
        message=req.message,
        without_memory=CompareSide(
            response=reply_bare,
            context_used=False,
            token_estimate=0,
            memory_facts=0,
        ),
        with_memory=CompareSide(
            response=reply_memory,
            context_used=bool(bundle.assembled_context),
            token_estimate=bundle.token_estimate,
            memory_facts=len(bundle.facts),
        ),
    )
