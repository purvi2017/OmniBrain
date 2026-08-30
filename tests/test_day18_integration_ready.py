"""
test_day18_integration_ready.py
---------------------------------
Day 18: Integration-Ready Final Version Test Suite.

Verifies:
1. Web UI Dashboard static asset serving (`/`, `/static/styles.css`, `/static/app.js`).
2. CORS middleware header configuration for external frontend/microservice integration.
3. Complete end-to-end integration flow across PDF Ingestion, Semantic Retrieval, LLM Grounding, and Multi-turn Chat.
4. Data contract fidelity across all Pydantic schemas.
5. Integration-ready error handling, session persistence, and state isolation.
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
from schemas import (
    BackendQueryInput,
    BackendQueryOutput,
    RAGQueryInput,
    RAGQueryOutput,
    SourceAttribution,
)

# Test PDF files
SAMPLE_PDF = str(BASE_DIR / "test_files" / "sample.pdf")
AI_PDF = str(BASE_DIR / "test_files" / "ai_architecture.pdf")
MARKET_PDF = str(BASE_DIR / "test_files" / "market_analysis_2026.pdf")


def mock_day18_llm(prompt: str) -> str:
    """Deterministic mock responder for Day 18 final integration-ready test suite."""
    prompt_lower = prompt.lower()

    if "unsupported" in prompt_lower or "alien" in prompt_lower:
        return NO_CONTEXT_MESSAGE
    if "omnibrain" in prompt_lower or "microservice" in prompt_lower:
        return "The OmniBrain system is composed of Document Parser, Vector Storage, and LLM Orchestrator."
    if "cyberpulse" in prompt_lower or "revenue" in prompt_lower:
        return "CyberPulse Systems reported revenue of $42.5 Million in Q3 2026."
    if "pypdf" in prompt_lower:
        return "pypdf is used for text extraction."

    return "Based on the retrieved document context, accurate details were verified."


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global app state prior to each test."""
    state = get_state()
    state.store = None
    state.sessions.clear()
    state.ingested_documents = []
    state.client = mock_day18_llm
    yield
    state.sessions.clear()


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


# ------------------------------------------------------------
# 1. Web UI Dashboard & Static Asset Tests
# ------------------------------------------------------------

def test_day18_ui_index_route(client):
    """Verify GET / serves the HTML Web UI dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "OmniBrain" in response.text
    assert "Integration Dashboard" in response.text


def test_day18_static_assets_serving(client):
    """Verify static styles.css and app.js assets are served with 200 OK."""
    res_css = client.get("/static/styles.css")
    assert res_css.status_code == 200
    assert "app-container" in res_css.text

    res_js = client.get("/static/app.js")
    assert res_js.status_code == 200
    assert "API_BASE" in res_js.text


# ------------------------------------------------------------
# 2. CORS Middleware Headers Verification
# ------------------------------------------------------------

def test_day18_cors_headers(client):
    """Verify CORS middleware headers are set for cross-origin requests."""
    response = client.options(
        "/query",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in ["*", "http://localhost:3000"]


# ------------------------------------------------------------
# 3. End-to-End Integration Pipeline Tests
# ------------------------------------------------------------

def test_day18_full_integration_workflow(client):
    """
    Test complete integration flow:
    Health Check -> Ingest PDFs -> Single Query -> Multi-turn Chat -> Session Inspection -> Session Clear.
    """
    # 1. Health check before ingestion
    h1 = client.get("/health").json()
    assert h1["indexed_vectors"] == 0

    # 2. Ingest PDFs
    with open(SAMPLE_PDF, "rb") as f1, open(AI_PDF, "rb") as f2:
        files = [
            ("files", ("sample.pdf", f1.read(), "application/pdf")),
            ("files", ("ai_architecture.pdf", f2.read(), "application/pdf")),
        ]
        ingest_res = client.post("/ingest", files=files)
    assert ingest_res.status_code == 200
    ingest_data = ingest_res.json()
    assert len(ingest_data["documents"]) == 2
    assert ingest_data["total_vectors"] > 0

    # 3. Health check after ingestion
    h2 = client.get("/health").json()
    assert h2["indexed_vectors"] > 0

    # 4. Single-turn grounded query
    q_res = client.post("/query", json={"query": "What text extraction library is used?"})
    assert q_res.status_code == 200
    q_data = q_res.json()
    assert q_data["found"] is True
    assert len(q_data["sources"]) > 0

    # 5. Multi-turn conversational chat session
    sid = "final_integration_session"
    c1 = client.post(f"/chat/{sid}", json={"message": "What microservices exist in OmniBrain?"})
    assert c1.status_code == 200
    assert c1.json()["turn"] == 1

    c2 = client.post(f"/chat/{sid}", json={"message": "Which service manages vector storage?"})
    assert c2.status_code == 200
    assert c2.json()["turn"] == 2

    # 6. Session registry inspection
    sessions_res = client.get("/sessions").json()
    assert sessions_res["total"] >= 1
    session_item = next(s for s in sessions_res["sessions"] if s["session_id"] == sid)
    assert session_item["turn_count"] == 2

    # 7. Session history retrieval
    hist_res = client.get(f"/sessions/{sid}").json()
    assert hist_res["session_id"] == sid
    assert len(hist_res["messages"]) == 4  # 2 turns x (user + assistant)

    # 8. Session clear
    del_res = client.delete(f"/sessions/{sid}")
    assert del_res.status_code == 200
    assert client.get(f"/sessions/{sid}").status_code == 404


# ------------------------------------------------------------
# 4. Data Contract Fidelity & Schema Verification
# ------------------------------------------------------------

def test_day18_schema_fidelity():
    """Verify BackendQueryInput and BackendQueryOutput Pydantic schema validation."""
    inp = BackendQueryInput(
        query="What is the revenue?",
        context="CyberPulse Systems recorded revenue of $42.5 Million in Q3 2026.",
        source_document="market_analysis_2026.pdf",
        document_id="market_analysis_2026",
    )
    assert inp.query == "What is the revenue?"
    assert inp.source_document == "market_analysis_2026.pdf"

    out = BackendQueryOutput(
        query="What is the revenue?",
        answer="CyberPulse Systems recorded revenue of $42.5 Million in Q3 2026.",
        source_document="market_analysis_2026.pdf",
        document_id="market_analysis_2026",
        relevant_context=inp.context,
        found=True,
        model="gemini-2.5-flash",
    )
    assert out.found is True
    assert out.query == "What is the revenue?"
    assert out.source_document == "market_analysis_2026.pdf"
    assert out.model_dump()["found"] is True
