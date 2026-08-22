"""
schemas.py
----------
Standardized Data Schemas and Interfaces for the RAG Pipeline.

Defines strongly-typed input and output structures for backend integration,
single-turn RAG queries, conversational chat turns, and source attributions.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ------------------------------------------------------------
# Source Citation / Attribution Model
# ------------------------------------------------------------

class SourceAttribution(BaseModel):
    """Represents precise document chunk attribution for retrieved context."""
    source: str = Field(..., description="Source file name or identifier")
    filename: str = Field(..., description="Original filename of the document")
    document_id: str = Field(..., description="Unique document ID (stem)")
    chunk_id: int = Field(..., description="0-indexed chunk identifier within document")
    page: int = Field(1, description="1-indexed page number where chunk resides")
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    confidence: str = Field("MEDIUM", description="Confidence label (HIGH, MEDIUM, LOW)")
    text_preview: str = Field("", description="Truncated text preview of retrieved chunk")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceAttribution":
        """Construct SourceAttribution safely from raw retrieval dictionary."""
        return cls(
            source=data.get("source", "unknown"),
            filename=data.get("filename", data.get("source", "unknown")),
            document_id=data.get("document_id", ""),
            chunk_id=data.get("chunk_id", -1),
            page=data.get("page", 1),
            score=round(float(data.get("score", 0.0)), 4),
            confidence=data.get("confidence", "MEDIUM"),
            text_preview=data.get("text_preview", data.get("text", "")[:150].replace("\n", " ")),
        )


# ------------------------------------------------------------
# RAG Query Input & Output Schemas
# ------------------------------------------------------------

class RAGQueryInput(BaseModel):
    """Input payload for a single-turn RAG query."""
    query: str = Field(..., min_length=1, description="Question string to answer from indexed documents")
    top_k: int = Field(3, ge=1, le=20, description="Maximum context chunks to retrieve")
    min_score: float = Field(0.35, ge=0.0, le=1.0, description="Minimum similarity score threshold")
    max_score_drop: float = Field(0.25, ge=0.0, le=1.0, description="Max relative drop allowed from top-1 score")
    document_id: Optional[str] = Field(None, description="Optional document ID constraint")


class RAGQueryOutput(BaseModel):
    """Output payload returned by a single-turn RAG query execution."""
    query: str = Field(..., description="The processed query string")
    answer: str = Field(..., description="Generated LLM response or refusal message")
    found: bool = Field(..., description="True if relevant document context was retrieved")
    sources: List[SourceAttribution] = Field(default_factory=list, description="Retrieved source attributions")
    retrieved_chunks: int = Field(0, description="Number of matching context chunks retrieved")
    model: str = Field(..., description="LLM model identifier used for generation")


# ------------------------------------------------------------
# Conversational Chat Input & Output Schemas
# ------------------------------------------------------------

class ChatTurnInput(BaseModel):
    """Input payload for a stateful conversational chat turn."""
    message: str = Field(..., min_length=1, description="User conversational message or follow-up question")
    top_k: int = Field(3, ge=1, le=20, description="Maximum context chunks to retrieve")
    min_score: float = Field(0.35, ge=0.0, le=1.0, description="Minimum similarity score threshold")
    max_score_drop: float = Field(0.25, ge=0.0, le=1.0, description="Max relative drop allowed from top-1 score")
    document_id: Optional[str] = Field(None, description="Optional document ID constraint")


class ChatTurnOutput(BaseModel):
    """Output payload returned by a conversational chat turn execution."""
    session_id: str = Field(..., description="Unique session identifier")
    turn: int = Field(..., description="Turn counter (1-indexed)")
    query: str = Field(..., description="User message for this turn")
    answer: str = Field(..., description="Generated assistant response")
    found: bool = Field(..., description="True if relevant document context was retrieved")
    sources: List[SourceAttribution] = Field(default_factory=list, description="Retrieved source attributions")
    retrieved_chunks: int = Field(0, description="Number of matching context chunks retrieved")
    model: str = Field(..., description="LLM model identifier used for generation")
