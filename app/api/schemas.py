from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=12)
    reranker_enabled: bool | None = None
    temperature: float | None = Field(default=None, ge=0.1, le=0.9)


class Citation(BaseModel):
    source: str
    page: int
    file_type: str
    similarity: float


class AskResponse(BaseModel):
    answer: str
    session_id: str
    citations: list[Citation]
    refused: bool
    refusal_reason: str | None = None
    latency_ms: int
    token_in: int
    token_out: int
    retrieved_chunks: list[dict[str, Any]]
    request_id: str


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    model_present: bool
    chunk_count: int = 0
