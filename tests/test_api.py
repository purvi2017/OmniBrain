"""
test_api.py
-----------
Day 10 Test Suite: FastAPI REST API for the RAG Pipeline.

Uses httpx AsyncClient with FastAPI's async test client.
The Gemini LLM client is replaced with a mock callable so no real API calls are made.

Verifies:
1.  GET  /health          — status, indexed_vectors, active_sessions fields
2.  POST /ingest          — PDF upload, schema validation, vector count
3.  POST /ingest          — non-PDF file rejected with 400
4.  POST /query           — single-turn response schema
5.  POST /query           — out-of-domain query returns found=False
6.  POST /query           — before ingestion returns 503
7.  POST /chat/{id}       — first turn creates session, turn=1
8.  POST /chat/{id}       — second turn increments to turn=2
9.  GET  /sessions        — lists active sessions with turn counts
10. GET  /sessions/{id}   — returns message history for session
11. DELETE /sessions/{id} — clears session, subsequent GET returns 404
12. DELETE /sessions/{id} — nonexistent session returns 404
"""

import os
import sys
from pathlib import Path
from typing import Any

import pytest
import httpx
from fastapi import status
from fastapi.testclient import TestClient

# Ensure project root is on PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import api as api_module
from api import app, get_state

SAMPLE_PDF = str(BASE_DIR / "test_files" / "sample.pdf")
AI_PDF = str(BASE_DIR / "test_files" / "ai_architecture.pdf")

# ---------------------------------------------------------------------------
# Mock LLM client — returns deterministic answers without calling Gemini
# ---------------------------------------------------------------------------

def _mock_llm(prompt: str) -> str:
    if "microservice" in prompt.lower() or "omnibrain" in prompt.lower():
        return "The OmniBrain architecture uses a Document Parser Service and an LLM Orchestrator."
    if "pypdf" in prompt.lower() or "extraction" in prompt.lower():
        return "pypdf is used for text extraction from the sample PDF."
    return "Based on the retrieved context, this is a test document."


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_state():
    """
    Reset global AppState before every test so tests are fully isolated.
    Injects the mock LLM client so no real Gemini calls are made.
    """
    state = get_state()
    state.store = None
    state.sessions.clear()
    state.ingested_documents = []
    state.client = _mock_llm       # inject mock
    state.model_name = "mock-gemini"
    yield
    # Cleanup after test
    state.store = None
    state.sessions.clear()


@pytest.fixture()
def client():
    """Synchronous TestClient for simple, non-async tests."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture()
def ingested_client():
    """
    TestClient with sample.pdf already ingested so retrieval tests can run
    without repeating the ingest step.
    """
    with TestClient(app, raise_server_exceptions=True) as c:
        with open(SAMPLE_PDF, "rb") as f:
            resp = c.post("/ingest", files=[("files", ("sample.pdf", f, "application/pdf"))])
        assert resp.status_code == 200, f"Ingest failed: {resp.text}"
        yield c


@pytest.fixture()
def multi_doc_client():
    """TestClient with both sample.pdf and ai_architecture.pdf ingested."""
    with TestClient(app, raise_server_exceptions=True) as c:
        with open(SAMPLE_PDF, "rb") as f1, open(AI_PDF, "rb") as f2:
            resp = c.post(
                "/ingest",
                files=[
                    ("files", ("sample.pdf", f1, "application/pdf")),
                    ("files", ("ai_architecture.pdf", f2, "application/pdf")),
                ],
            )
        assert resp.status_code == 200, f"Ingest failed: {resp.text}"
        yield c


# ---------------------------------------------------------------------------
# 1. GET /health — before and after ingestion
# ---------------------------------------------------------------------------

def test_health_before_ingest(client):
    """Health endpoint returns ok with 0 indexed vectors before any ingest."""
    print("=" * 60)
    print("TEST: GET /health (before ingest)")
    print("=" * 60)

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["indexed_vectors"] == 0
    assert body["active_sessions"] == 0
    assert "model" in body
    print(f"[PASS] /health → status=ok, indexed_vectors=0, model={body['model']}")


def test_health_after_ingest(ingested_client):
    """Health endpoint shows indexed_vectors > 0 after successful ingest."""
    print("=" * 60)
    print("TEST: GET /health (after ingest)")
    print("=" * 60)

    resp = ingested_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["indexed_vectors"] > 0
    print(f"[PASS] /health → indexed_vectors={body['indexed_vectors']}")


# ---------------------------------------------------------------------------
# 2 & 3. POST /ingest
# ---------------------------------------------------------------------------

def test_ingest_single_pdf(client):
    """POST /ingest with a single PDF returns correct IngestResponse schema."""
    print("=" * 60)
    print("TEST: POST /ingest — single PDF")
    print("=" * 60)

    assert os.path.exists(SAMPLE_PDF)
    with open(SAMPLE_PDF, "rb") as f:
        resp = client.post(
            "/ingest",
            files=[("files", ("sample.pdf", f, "application/pdf"))],
        )

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert "message" in body
    assert "documents" in body
    assert len(body["documents"]) == 1
    assert body["documents"][0]["document_id"] == "sample"
    assert body["total_chunks"] > 0
    assert body["total_vectors"] > 0
    print(
        f"[PASS] Ingested 1 PDF → "
        f"{body['total_chunks']} chunks, {body['total_vectors']} vectors"
    )


def test_ingest_multiple_pdfs(client):
    """POST /ingest with multiple PDFs indexes all of them."""
    print("=" * 60)
    print("TEST: POST /ingest — multiple PDFs")
    print("=" * 60)

    assert os.path.exists(SAMPLE_PDF)
    assert os.path.exists(AI_PDF)

    with open(SAMPLE_PDF, "rb") as f1, open(AI_PDF, "rb") as f2:
        resp = client.post(
            "/ingest",
            files=[
                ("files", ("sample.pdf", f1, "application/pdf")),
                ("files", ("ai_architecture.pdf", f2, "application/pdf")),
            ],
        )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["documents"]) == 2
    doc_ids = {d["document_id"] for d in body["documents"]}
    assert doc_ids == {"sample", "ai_architecture"}
    print(f"[PASS] Ingested 2 PDFs → {body['total_vectors']} total vectors")


def test_ingest_non_pdf_rejected(client):
    """POST /ingest with a non-PDF file returns 400 Bad Request."""
    print("=" * 60)
    print("TEST: POST /ingest — non-PDF file rejected")
    print("=" * 60)

    resp = client.post(
        "/ingest",
        files=[("files", ("notes.txt", b"hello world", "text/plain"))],
    )
    assert resp.status_code == 400
    assert "PDF" in resp.json()["detail"]
    print("[PASS] Non-PDF upload correctly rejected with 400")


# ---------------------------------------------------------------------------
# 4 & 5. POST /query
# ---------------------------------------------------------------------------

def test_query_response_schema(ingested_client):
    """POST /query returns a correctly structured QueryResponse."""
    print("=" * 60)
    print("TEST: POST /query — response schema")
    print("=" * 60)

    resp = ingested_client.post(
        "/query",
        json={"query": "What library is used for text extraction?", "min_score": 0.20},
    )

    assert resp.status_code == 200, f"Unexpected status: {resp.status_code}: {resp.text}"
    body = resp.json()
    assert "query" in body
    assert "answer" in body
    assert "found" in body
    assert "sources" in body
    assert "retrieved_chunks" in body
    assert "model" in body
    assert body["query"] == "What library is used for text extraction?"
    print(
        f"[PASS] /query → found={body['found']}, "
        f"retrieved_chunks={body['retrieved_chunks']}, "
        f"sources={len(body['sources'])}"
    )

    if body["found"]:
        src = body["sources"][0]
        assert "source" in src
        assert "filename" in src
        assert "page" in src
        assert "score" in src
        assert "confidence" in src
        assert "text_preview" in src
        assert isinstance(src["score"], float)
        print("[PASS] Source citation structure validated")


def test_query_out_of_domain(ingested_client):
    """POST /query with a high threshold for an unrelated query returns found=False."""
    print("=" * 60)
    print("TEST: POST /query — out-of-domain query")
    print("=" * 60)

    resp = ingested_client.post(
        "/query",
        json={
            "query": "What is the melting point of osmium?",
            "min_score": 0.90,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False
    assert body["retrieved_chunks"] == 0
    assert body["sources"] == []
    print("[PASS] Out-of-domain query correctly returns found=False with empty sources")


def test_query_before_ingest_returns_503(client):
    """POST /query before any ingestion returns HTTP 503."""
    print("=" * 60)
    print("TEST: POST /query — before ingest → 503")
    print("=" * 60)

    resp = client.post("/query", json={"query": "What is this?"})
    assert resp.status_code == 503
    assert "ingest" in resp.json()["detail"].lower()
    print("[PASS] /query before ingest returns 503 with helpful message")


# ---------------------------------------------------------------------------
# 7 & 8. POST /chat/{session_id}
# ---------------------------------------------------------------------------

def test_chat_first_turn(ingested_client):
    """POST /chat/{id} creates a session and returns turn=1."""
    print("=" * 60)
    print("TEST: POST /chat — first turn")
    print("=" * 60)

    resp = ingested_client.post(
        "/chat/test-session-001",
        json={"message": "What is the purpose of this document?", "min_score": 0.20},
    )

    assert resp.status_code == 200, f"Status: {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["session_id"] == "test-session-001"
    assert body["turn"] == 1
    assert "answer" in body
    assert "found" in body
    assert "sources" in body
    assert "retrieved_chunks" in body
    print(
        f"[PASS] First chat turn → session_id={body['session_id']}, "
        f"turn={body['turn']}, found={body['found']}"
    )


def test_chat_second_turn_increments(ingested_client):
    """POST /chat/{id} on second message increments turn to 2."""
    print("=" * 60)
    print("TEST: POST /chat — second turn increments")
    print("=" * 60)

    ingested_client.post(
        "/chat/test-session-002",
        json={"message": "What library is used for text extraction?", "min_score": 0.20},
    )
    resp = ingested_client.post(
        "/chat/test-session-002",
        json={"message": "Tell me more about how it extracts text.", "min_score": 0.20},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "test-session-002"
    assert body["turn"] == 2
    print(f"[PASS] Second turn → turn={body['turn']}")


def test_chat_before_ingest_returns_503(client):
    """POST /chat before ingestion returns 503."""
    print("=" * 60)
    print("TEST: POST /chat — before ingest → 503")
    print("=" * 60)

    resp = client.post("/chat/no-docs", json={"message": "Hello?"})
    assert resp.status_code == 503
    print("[PASS] /chat before ingest returns 503")


# ---------------------------------------------------------------------------
# 9. GET /sessions
# ---------------------------------------------------------------------------

def test_list_sessions(ingested_client):
    """GET /sessions lists sessions with correct turn counts."""
    print("=" * 60)
    print("TEST: GET /sessions")
    print("=" * 60)

    # Create two sessions
    ingested_client.post(
        "/chat/sess-alpha",
        json={"message": "What is FAISS?", "min_score": 0.20},
    )
    ingested_client.post(
        "/chat/sess-beta",
        json={"message": "Describe the chunker.", "min_score": 0.20},
    )

    resp = ingested_client.get("/sessions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2

    session_ids = {s["session_id"] for s in body["sessions"]}
    assert "sess-alpha" in session_ids
    assert "sess-beta" in session_ids

    for s in body["sessions"]:
        assert "turn_count" in s
        assert "message_count" in s

    print(f"[PASS] /sessions → {body['total']} sessions listed correctly")


# ---------------------------------------------------------------------------
# 10. GET /sessions/{session_id}
# ---------------------------------------------------------------------------

def test_get_session_history(ingested_client):
    """GET /sessions/{id} returns the full message history."""
    print("=" * 60)
    print("TEST: GET /sessions/{session_id}")
    print("=" * 60)

    ingested_client.post(
        "/chat/hist-session",
        json={"message": "What library is used for text extraction?", "min_score": 0.20},
    )

    resp = ingested_client.get("/sessions/hist-session")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "hist-session"
    assert body["turn_count"] == 1
    assert len(body["messages"]) == 2  # one user + one assistant message

    user_msg = body["messages"][0]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == "What library is used for text extraction?"
    assert "timestamp" in user_msg

    assistant_msg = body["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert "content" in assistant_msg

    print(
        f"[PASS] /sessions/hist-session → "
        f"turn_count={body['turn_count']}, messages={len(body['messages'])}"
    )


def test_get_session_not_found(ingested_client):
    """GET /sessions/{nonexistent_id} returns 404."""
    print("=" * 60)
    print("TEST: GET /sessions/{nonexistent_id} → 404")
    print("=" * 60)

    resp = ingested_client.get("/sessions/does-not-exist")
    assert resp.status_code == 404
    print("[PASS] Nonexistent session GET returns 404")


# ---------------------------------------------------------------------------
# 11 & 12. DELETE /sessions/{session_id}
# ---------------------------------------------------------------------------

def test_delete_session(ingested_client):
    """DELETE /sessions/{id} clears the session; subsequent GET returns 404."""
    print("=" * 60)
    print("TEST: DELETE /sessions/{session_id}")
    print("=" * 60)

    # Create a session
    ingested_client.post(
        "/chat/del-session",
        json={"message": "What is chunking?", "min_score": 0.20},
    )

    # Verify it exists
    resp = ingested_client.get("/sessions/del-session")
    assert resp.status_code == 200

    # Delete it
    del_resp = ingested_client.delete("/sessions/del-session")
    assert del_resp.status_code == 200
    assert "cleared" in del_resp.json()["message"].lower()
    print("[PASS] DELETE /sessions/del-session → 200 with confirmation")

    # Verify it's gone
    resp = ingested_client.get("/sessions/del-session")
    assert resp.status_code == 404
    print("[PASS] Subsequent GET after DELETE → 404")


def test_delete_session_not_found(ingested_client):
    """DELETE /sessions/{nonexistent_id} returns 404."""
    print("=" * 60)
    print("TEST: DELETE /sessions/{nonexistent_id} → 404")
    print("=" * 60)

    resp = ingested_client.delete("/sessions/ghost-session")
    assert resp.status_code == 404
    print("[PASS] Nonexistent session DELETE returns 404")


# ---------------------------------------------------------------------------
# Run all tests manually
# ---------------------------------------------------------------------------

def run_all_tests():
    """Manual test runner (non-pytest)."""
    print("=" * 70)
    print("RUNNING DAY 10 API TEST SUITE (manual)")
    print("=" * 70)
    print("Run with pytest instead: pytest tests/test_api.py -v")


if __name__ == "__main__":
    run_all_tests()
