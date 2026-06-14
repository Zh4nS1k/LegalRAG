"""Pydantic schemas for the Legal RAG Tester project."""
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Optional

class Question(BaseModel):
    """Represents a legal question read from the input Excel."""
    id: str
    text: str

class Chunk(BaseModel):
    """Represents a document chunk retrieved from Pinecone."""
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)

class LLMResult(BaseModel):
    """Represents the execution result of a single LLM call."""
    model: str
    answer: str = ""
    answer_raw: str = ""
    error: str = ""
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0
    tokens_per_sec: float = 0.0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    chunks_used: int = 0
    avg_score: float = 0.0
    quality_score: Optional[int] = None   # None=scorer failed, 0=truly bad answer
    quality_reason: str = ""
    quality_rank: int = 0
    retrieved_scores: list[float] = Field(default_factory=list)

class TestRow(BaseModel):
    """Represents a combination of Question and its corresponding LLMResult."""
    question: Question
    result: LLMResult
