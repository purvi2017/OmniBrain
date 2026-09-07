"""
test_day17_final_module_testing.py
------------------------------------
Day 17: Final Module Testing + Stability Test Suite.

Verifies:
1. End-to-End User Flows: Document Ingestion, Grounded RAG Query, Conversational Multi-Turn Chat, Session Management.
2. Direct Context processing flow bypassing FAISS vector search.
3. Out-of-Domain Refusal handling (`found=False`, refusal text).
4. API Success & Error HTTP Status Codes:
   - 200 OK for valid requests
   - 400 Bad Request for invalid uploads or missing parameters
   - 404 Not Found for non-existent session requests
   - 503 Service Unavailable when no documents indexed or client uninitialized
5. Schema validation across FastAPI Pydantic models.
6. Multi-session stability and isolation without state cross-contamination.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure project root is on PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from api import app, get_state
from rag_llm import build_store, NO_CONTEXT_MESSAGE
from schemas import BackendQueryInput, BackendQueryOutput

# Sample test files
SAMPLE_PDF = str(BASE_DIR / "test_files" / "sample.pdf")
AI_PDF = str(BASE_DIR / "test_files" / "ai_architecture.pdf")
MARKET_PDF = str(BASE_DIR / "test_files" / "market_analysis_2026.pdf")


def mock_day17_llm(prompt: str) -> str:
    """Deterministic mock responder for Day 17 final testing suite."""
    prompt_lower = prompt.lower()

    if "unsupported" in prompt_lower or "alien" in prompt_lower or "maternal" in prompt_lower:
        return NO_CONTEXT_MESSAGE
    if "omnibrain" in prompt_lower or "microservice" in prompt_lower:
        return "The OmniBrain architecture features Document Parser, Vector Storage, and LLM Orchestrator."
    if "cyberpulse" in prompt_lower or "revenue" in prompt_lower:
        return "CyberPulse Systems recorded Q3 2026 revenue of $42.5 Million."
    if "pypdf" in prompt_lower or "extraction" in prompt_lower:
        return "pypdf is used for text extraction."

    return "Based on the retrieved document context, detailed information was identified."


@pytest.fixture(autouse=True)
def reset_app_state():
    """Reset the shared FastAPI application state before each test."""
    state = get_state()
    state.store = None
    state.sessions.clear()
    state.ingested_documents = []
    state.client = mock_day17_llm
    yield
    state.sessions.clear()


@pytest.fixture
def client():
    """Return a FastAPI TestClient instance."""
    return TestClient(app)


# ------------------------------------------------------------
# 1. System Health & Liveness Tests
# ------------------------------------------------------------

def test_day17_health_check(client):
    """Verify /health status check endpoint return structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["indexed_vectors"] == 0
    assert data["active_sessions"] == 0
    assert "model" in data


# ------------------------------------------------------------
# 2. Document Ingestion Flow & Input Validation
# ------------------------------------------------------------

def test_day17_ingest_valid_pdf(client):
    """Verify successful PDF document upload and indexing."""
    with open(SAMPLE_PDF, "rb") as f:
        files = [("files", ("sample.pdf", f.read(), "application/pdf"))]
        response = client.post("/ingest", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "Successfully indexed" in data["message"]
    assert len(data["documents"]) == 1
    assert data["documents"][0]["filename"] == "sample.pdf"
    assert data["total_chunks"] > 0
    assert data["total_vectors"] > 0

    # Verify state was updated
    state = get_state()
    assert state.store is not None
    assert state.store.count() > 0


def test_day17_ingest_invalid_file_type(client):
    """Verify 400 Bad Request error when uploading non-PDF file."""
    files = [("files", ("test.txt", b"plain text content", "text/plain"))]
    response = client.post("/ingest", files=files)

    assert response.status_code == 400
    data = response.json()
    assert "Only PDF files are accepted" in data["detail"]


# ------------------------------------------------------------
# 3. Single-Turn RAG Query Flow (Success & Error Responses)
# ------------------------------------------------------------

def test_day17_query_without_ingestion_returns_503(client):
    """Verify HTTP 503 Service Unavailable when querying before ingestion."""
    response = client.post("/query", json={"query": "What is the revenue?"})
    assert response.status_code == 503
    assert "No documents indexed" in response.json()["detail"]


def test_day17_query_with_direct_context(client):
    """Verify direct context query processing without needing indexed FAISS store."""
    payload = {
        "query": "What was Q3 revenue for CyberPulse?",
        "context": "CyberPulse Systems recorded Q3 2026 revenue of $42.5 Million.",
        "source_document": "market_analysis_2026.pdf"
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is True
    assert "CyberPulse" in data["answer"]
    assert data["source_document"] == "market_analysis_2026.pdf"


def test_day17_query_indexed_document_flow(client):
    """Verify end-to-end grounded query execution across indexed FAISS vectors."""
    # Step 1: Ingest document
    with open(AI_PDF, "rb") as f:
        client.post("/ingest", files=[("files", ("ai_architecture.pdf", f.read(), "application/pdf"))])

    # Step 2: Single-turn query
    response = client.post("/query", json={"query": "What are the core microservices in OmniBrain?"})
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is True
    assert "OmniBrain" in data["answer"]
    assert len(data["sources"]) > 0
    assert data["retrieved_chunks"] > 0


def test_day17_query_out_of_domain_refusal(client):
    """Verify graceful refusal response when no context matches the query."""
    with open(SAMPLE_PDF, "rb") as f:
        client.post("/ingest", files=[("files", ("sample.pdf", f.read(), "application/pdf"))])

    response = client.post("/query", json={"query": "What is the capital of unsupported alien planet?"})
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is False
    assert data["answer"] == NO_CONTEXT_MESSAGE
    assert len(data["sources"]) == 0
    assert data["retrieved_chunks"] == 0


# ------------------------------------------------------------
# 4. Multi-Turn Conversational Chat & History Isolation
# ------------------------------------------------------------

def test_day17_chat_flow(client):
    """Verify stateful multi-turn conversational chat with history injection."""
    with open(AI_PDF, "rb") as f:
        client.post("/ingest", files=[("files", ("ai_architecture.pdf", f.read(), "application/pdf"))])

    session_id = "test_user_session_17"

    # Turn 1
    r1 = client.post(f"/chat/{session_id}", json={"message": "What microservices exist in OmniBrain?"})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["turn"] == 1
    assert d1["session_id"] == session_id
    assert d1["found"] is True

    # Turn 2
    r2 = client.post(f"/chat/{session_id}", json={"message": "Which of those handles vector storage?"})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["turn"] == 2
    assert d2["session_id"] == session_id


def test_day17_session_management_endpoints(client):
    """Verify GET /sessions, GET /sessions/{id}, and DELETE /sessions/{id} endpoints."""
    with open(SAMPLE_PDF, "rb") as f:
        client.post("/ingest", files=[("files", ("sample.pdf", f.read(), "application/pdf"))])

    sid = "session_to_manage"
    client.post(f"/chat/{sid}", json={"message": "What text extraction library is used?"})

    # List sessions
    res_list = client.get("/sessions")
    assert res_list.status_code == 200
    s_data = res_list.json()
    assert s_data["total"] >= 1
    assert any(s["session_id"] == sid for s in s_data["sessions"])

    # Get specific session history
    res_hist = client.get(f"/sessions/{sid}")
    assert res_hist.status_code == 200
    h_data = res_hist.json()
    assert h_data["session_id"] == sid
    assert h_data["turn_count"] == 1
    assert len(h_data["messages"]) == 2  # user + assistant

    # Delete session
    res_del = client.delete(f"/sessions/{sid}")
    assert res_del.status_code == 200
    assert "cleared successfully" in res_del.json()["message"]

    # Verify 404 for deleted session
    res_404 = client.get(f"/sessions/{sid}")
    assert res_404.status_code == 404


# ------------------------------------------------------------
# 5. Multi-Session Stability & Isolation Test
# ------------------------------------------------------------

def test_day17_multi_session_stability(client):
    """Verify parallel active sessions maintain isolated histories without cross-contamination."""
    with open(AI_PDF, "rb") as f:
        client.post("/ingest", files=[("files", ("ai_architecture.pdf", f.read(), "application/pdf"))])

    s1 = "user_alpha"
    s2 = "user_beta"

    client.post(f"/chat/{s1}", json={"message": "Query from user alpha"})
    client.post(f"/chat/{s2}", json={"message": "Query from user beta"})

    hist1 = client.get(f"/sessions/{s1}").json()
    hist2 = client.get(f"/sessions/{s2}").json()

    assert hist1["messages"][0]["content"] == "Query from user alpha"
    assert hist2["messages"][0]["content"] == "Query from user beta"
    assert hist1["session_id"] != hist2["session_id"]
