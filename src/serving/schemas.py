from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question about the filing excerpt")
    context: str = Field(..., min_length=1, description="Filing excerpt to ground the answer in")
    max_new_tokens: Optional[int] = Field(None, ge=1, le=512)


class GenerateResponse(BaseModel):
    answer: str
    latency_ms: float
    numeric_hallucination_rate: float


class HealthResponse(BaseModel):
    status: str
    device: str
    model_id: str
    adapter_loaded: bool
