"""
api.py
------
Day 10: FastAPI REST API for the Multi-Document Conversational RAG Pipeline.

Exposes the complete RAG system built in Days 1–9 as a deployable HTTP microservice.

Endpoints:
    GET  /health                  — Liveness check: status, indexed vectors, sessions
    POST /ingest                  — Upload & index one or more PDF files into FAISS
    POST /query                   — Single-turn grounded RAG query (no history)
    POST /chat/{session_id}       — Conversational turn with persistent chat history
    GET  /sessions                — List active session IDs and turn counts
    GET  /sessions/{session_id}   — Full message history for a session
    DELETE /sessions/{session_id} — Clear a session's history

Usage:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Environment:
    GEMINI_API_KEY — required for LLM answer generation.
    GEMINI_MODEL   — optional, defaults to gemini-2.5-flash.
"""

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_chat import ConversationalRAGPipeline, ChatMessage
from rag_llm import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MIN_SCORE,
    DEFAULT_MAX_SCORE_DROP,
    DEFAULT_TOP_K,
    answer_query,
    retrieve_context,
    prepare_context,
    build_store,
    create_gemini_client,
    NO_CONTEXT_MESSAGE,
)


# ------------------------------------------------------------
# Application State
# ------------------------------------------------------------

class AppState:
    """Holds shared pipeline state for the FastAPI application."""

    def __init__(self):
        # Shared FAISS store (populated via /ingest)
        self.store = None
        # In-memory session registry: session_id -> ConversationalRAGPipeline
        self.sessions: Dict[str, ConversationalRAGPipeline] = {}
        # Shared Gemini client (or injectable mock for tests)
        self.client: Optional[Any] = None
        # Model name
        self.model_name: str = DEFAULT_GEMINI_MODEL
        # Documents ingested so far
        self.ingested_documents: List[Dict] = []


# Global state — one instance per process
_state = AppState()


def get_state() -> AppState:
    """Return the global application state."""
    return _state


# ------------------------------------------------------------
# Lifespan (startup / shutdown)
# ------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup; clean up on shutdown."""
    state = get_state()
    state.model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    # Try to initialize the Gemini client if the API key is present
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            state.client = create_gemini_client(api_key=api_key)
            print(f"[INFO] Gemini client initialized (model={state.model_name})")
        except Exception as exc:
            print(f"[WARN] Gemini client init failed: {exc}. /query and /chat will use mock mode.")
    else:
        print("[WARN] GEMINI_API_KEY not set. /query and /chat require a client or mock injection.")

    yield

    # Shutdown: nothing persistent to close for in-memory FAISS
    print("[INFO] API shutting down.")


# ------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------

app = FastAPI(
    title="RAG Pipeline API",
    description=(
        "Day 10: REST API for the multi-document Conversational RAG pipeline. "
        "Supports PDF ingestion, single-turn Q&A, and stateful multi-turn chat."
    ),
    version="10.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------
# Pydantic Schemas
# ------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    indexed_vectors: int
    active_sessions: int
    model: str


class IngestResponse(BaseModel):
    message: str
    documents: List[Dict]
    total_chunks: int
    total_vectors: int


class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to answer from the indexed documents")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20, description="Max chunks to retrieve")
    min_score: float = Field(DEFAULT_MIN_SCORE, ge=0.0, le=1.0, description="Minimum similarity score")
    max_score_drop: float = Field(DEFAULT_MAX_SCORE_DROP, ge=0.0, le=1.0)
    document_id: Optional[str] = Field(None, description="Constrain query to one document")


class SourceItem(BaseModel):
    source: str
    filename: str
    document_id: str
    chunk_id: int
    page: int
    score: float
    confidence: str
    text_preview: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    found: bool
    sources: List[SourceItem]
    retrieved_chunks: int
    model: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's conversational message or follow-up")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)
    min_score: float = Field(DEFAULT_MIN_SCORE, ge=0.0, le=1.0)
    max_score_drop: float = Field(DEFAULT_MAX_SCORE_DROP, ge=0.0, le=1.0)
    document_id: Optional[str] = None


class ChatResponse(BaseModel):
    query: str
    answer: str
    found: bool
    sources: List[SourceItem]
    retrieved_chunks: int
    model: str
    turn: int
    session_id: str


class SessionInfo(BaseModel):
    session_id: str
    turn_count: int
    message_count: int


class SessionListResponse(BaseModel):
    sessions: List[SessionInfo]
    total: int


class MessageItem(BaseModel):
    role: str
    content: str
    timestamp: str
    sources: List[Dict]


class SessionHistoryResponse(BaseModel):
    session_id: str
    turn_count: int
    messages: List[MessageItem]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _require_store():
    """Raise 503 if no documents have been ingested yet."""
    state = get_state()
    if state.store is None or state.store.count() == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No documents indexed. POST one or more PDF files to /ingest first."
            ),
        )


def _require_client():
    """Raise 503 if no LLM client is available."""
    state = get_state()
    if state.client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "LLM client not initialized. "
                "Set GEMINI_API_KEY environment variable and restart the server."
            ),
        )


def _build_source_items(raw_sources: List[Dict]) -> List[SourceItem]:
    """Convert raw source dicts from the pipeline into SourceItem models."""
    items = []
    for s in raw_sources:
        items.append(SourceItem(
            source=s.get("source", "unknown"),
            filename=s.get("filename", s.get("source", "unknown")),
            document_id=s.get("document_id", ""),
            chunk_id=s.get("chunk_id", -1),
            page=s.get("page", 1),
            score=round(float(s.get("score", 0.0)), 4),
            confidence=s.get("confidence", "MEDIUM"),
            text_preview=s.get("text_preview", s.get("text", "")[:150]),
        ))
    return items


def _get_or_create_session(session_id: str) -> ConversationalRAGPipeline:
    """Return an existing pipeline session or create a new one."""
    state = get_state()
    if session_id not in state.sessions:
        pipeline = ConversationalRAGPipeline(
            client=state.client,
            model_name=state.model_name,
            session_id=session_id,
        )
        # Share the already-built FAISS store
        pipeline._pipeline.store = state.store
        state.sessions[session_id] = pipeline
    return state.sessions[session_id]


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    tags=["Admin"],
)
async def health():
    """Return service status, number of indexed vectors, and active session count."""
    state = get_state()
    indexed = state.store.count() if state.store is not None else 0
    return HealthResponse(
        status="ok",
        indexed_vectors=indexed,
        active_sessions=len(state.sessions),
        model=state.model_name,
    )


@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and index PDF documents",
    tags=["Documents"],
)
async def ingest(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF files. Each file is saved to a temporary directory,
    processed (extract → chunk → embed), and indexed into the shared FAISS store.
    The previous store is replaced with the newly indexed one.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one PDF file must be uploaded.",
        )

    state = get_state()
    tmpdir = tempfile.mkdtemp(prefix="rag_ingest_")

    try:
        pdf_paths = []
        for upload in files:
            if not upload.filename or not upload.filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Only PDF files are accepted. Got: {upload.filename!r}",
                )
            dest = os.path.join(tmpdir, upload.filename)
            with open(dest, "wb") as f:
                content = await upload.read()
                f.write(content)
            pdf_paths.append(dest)

        # Build FAISS store from all uploaded PDFs
        store, metadata = build_store(pdf_paths=pdf_paths)

        # Update shared state
        state.store = store
        state.ingested_documents = metadata

        # Update all existing sessions to point to the new store
        for pipeline in state.sessions.values():
            pipeline._pipeline.store = store

        return IngestResponse(
            message=f"Successfully indexed {len(metadata)} document(s).",
            documents=metadata,
            total_chunks=sum(m["chunk_count"] for m in metadata),
            total_vectors=store.count(),
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Single-turn RAG query (no history)",
    tags=["Query"],
)
async def query(req: QueryRequest):
    """
    Execute a single-turn grounded RAG query across indexed documents.
    No conversation history is used. For stateful multi-turn chat, use /chat/{session_id}.
    """
    _require_store()
    _require_client()

    state = get_state()

    result = answer_query(
        client=state.client,
        store=state.store,
        query=req.query,
        top_k=req.top_k,
        min_score=req.min_score,
        max_score_drop=req.max_score_drop,
        document_id=req.document_id,
        model_name=state.model_name,
    )

    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        found=result["found"],
        sources=_build_source_items(result.get("sources", [])),
        retrieved_chunks=result["retrieved_chunks"],
        model=result["model"],
    )


@app.post(
    "/chat/{session_id}",
    response_model=ChatResponse,
    summary="Conversational turn with chat history",
    tags=["Chat"],
)
async def chat(session_id: str, req: ChatRequest):
    """
    Send a message in a conversational session. Prior turns for this session_id
    are injected into the LLM prompt, enabling natural follow-up questions.
    A new session is created automatically if session_id has not been used before.
    """
    _require_store()
    _require_client()

    pipeline = _get_or_create_session(session_id)

    result = pipeline.chat(
        message=req.message,
        top_k=req.top_k,
        min_score=req.min_score,
        max_score_drop=req.max_score_drop,
        document_id=req.document_id,
    )

    return ChatResponse(
        query=result["query"],
        answer=result["answer"],
        found=result["found"],
        sources=_build_source_items(result.get("sources", [])),
        retrieved_chunks=result["retrieved_chunks"],
        model=result["model"],
        turn=result["turn"],
        session_id=result["session_id"],
    )


@app.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="List active sessions",
    tags=["Sessions"],
)
async def list_sessions():
    """Return a list of all active session IDs along with their turn counts."""
    state = get_state()
    sessions = [
        SessionInfo(
            session_id=sid,
            turn_count=pipeline.session.turn_count(),
            message_count=len(pipeline.session.messages),
        )
        for sid, pipeline in state.sessions.items()
    ]
    return SessionListResponse(sessions=sessions, total=len(sessions))


@app.get(
    "/sessions/{session_id}",
    response_model=SessionHistoryResponse,
    summary="Get session message history",
    tags=["Sessions"],
)
async def get_session(session_id: str):
    """Return the full message history for a given session."""
    state = get_state()
    if session_id not in state.sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    pipeline = state.sessions[session_id]
    messages = [
        MessageItem(
            role=m.role,
            content=m.content,
            timestamp=m.timestamp,
            sources=m.sources,
        )
        for m in pipeline.session.messages
    ]
    return SessionHistoryResponse(
        session_id=session_id,
        turn_count=pipeline.session.turn_count(),
        messages=messages,
    )


@app.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Clear session history",
    tags=["Sessions"],
)
async def delete_session(session_id: str):
    """
    Clear all conversation history for a session.
    The session object is removed from the registry entirely.
    """
    state = get_state()
    if session_id not in state.sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    del state.sessions[session_id]
    return {"message": f"Session '{session_id}' cleared successfully."}


# ------------------------------------------------------------
# Dev Entry Point
# ------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
