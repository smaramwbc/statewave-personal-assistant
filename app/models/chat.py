"""Request / response models for the chat API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(..., examples=["dev_alice", "dev_bob"])
    message: str = Field(..., min_length=1, max_length=4096)


class ChatResponse(BaseModel):
    user_id: str
    message: str
    response: str
    episode_id: str
    context_used: bool
    token_estimate: int


class CompareSide(BaseModel):
    response: str
    context_used: bool
    token_estimate: int
    memory_facts: int


class CompareResponse(BaseModel):
    user_id: str
    message: str
    without_memory: CompareSide
    with_memory: CompareSide
