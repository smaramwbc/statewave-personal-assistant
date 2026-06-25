"""Shared pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)



def make_statewave_context_response(
    subject_id: str = "user_1",
    assembled_context: str = "User is a senior engineer. Prefers Python.",
    token_estimate: int = 42,
    facts: list | None = None,
) -> dict:
    return {
        "subject_id": subject_id,
        "assembled_context": assembled_context,
        "token_estimate": token_estimate,
        "facts": facts
        or [
            {
                "id": "mem_001",
                "subject_id": subject_id,
                "kind": "profile_fact",
                "content": "Senior engineer, enterprise plan.",
                "confidence": 0.98,
                "source_episode_ids": ["ep_001"],
                "created_at": "2024-06-01T10:00:00Z",
                "tags": ["role"],
            }
        ],
    }


def make_statewave_episode_response(episode_id: str = "ep_new") -> dict:
    return {
        "id": episode_id,
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


def make_statewave_memory_search(subject_id: str = "user_1") -> dict:
    return {
        "memories": [
            {
                "id": "mem_001",
                "subject_id": subject_id,
                "kind": "profile_fact",
                "content": "Senior engineer.",
                "confidence": 1.0,
                "source_episode_ids": ["ep_001"],
                "created_at": "2024-06-01T10:00:00Z",
                "tags": ["role"],
            },
            {
                "id": "mem_002",
                "subject_id": subject_id,
                "kind": "episode_summary",
                "content": "Prefers Python.",
                "confidence": 0.95,
                "source_episode_ids": ["ep_001"],
                "created_at": "2024-06-01T10:00:00Z",
                "tags": ["language"],
            },
        ]
    }
