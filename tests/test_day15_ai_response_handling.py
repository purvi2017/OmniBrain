"""
test_day15_ai_response_handling.py
-----------------------------------
Day 15: AI Response Handling + Backend Integration Refinement Test Suite.

Verifies:
1. Processing of query and relevant document context received from Backend.
2. Generation of grounded answers from available document context.
3. Handling of missing, empty, or incomplete context scenarios.
4. Consistent structured response payloads containing:
   - answer
   - source_document
   - document_id
   - relevant_context
5. Multiple query types using real document content (Healthcare, Finance, Tech Spec).
6. Appropriate fallback responses for irrelevant or unsupported queries.
7. Error and failure scenarios (empty queries, missing client/store, LLM exceptions).
8. Validation against Pydantic BackendQueryInput and BackendQueryOutput schemas.
9. FastAPI REST endpoint POST /query backend integration verification.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure project root is on PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag_llm import (
    process_backend_query,
    build_store,
    answer_query,
    NO_CONTEXT_MESSAGE,
    RAGLLMPipeline,
)
from schemas import BackendQueryInput, BackendQueryOutput, SourceAttribution
from api import app, get_state

# Real test PDFs
SAMPLE_PDF = str(BASE_DIR / "test_files" / "sample.pdf")
AI_PDF = str(BASE_DIR / "test_files" / "ai_architecture.pdf")
MARKET_PDF = str(BASE_DIR / "test_files" / "market_analysis_2026.pdf")
MULTIPAGE_PDF = str(BASE_DIR / "test_files" / "multipage_manual.pdf")


def mock_day15_llm(prompt: str) -> str:
    """Deterministic mock responder for Day 15 AI response handling tests."""
    prompt_lower = prompt.lower()

    if "unsupported" in prompt_lower or "alien" in prompt_lower or "maternal" in prompt_lower:
        return NO_CONTEXT_MESSAGE
    if "omnibrain" in prompt_lower or "microservice" in prompt_lower:
        return "The OmniBrain architecture contains Document Parser Service and LLM Orchestrator."
    if "cyberpulse" in prompt_lower:
        return "CyberPulse Systems recorded revenue of $42.5 Million in Q3 2026 with a net margin of 24%."
    if "pypdf" in prompt_lower or "extraction" in prompt_lower:
        return "pypdf is used for text extraction."
    if "dental" in prompt_lower or "$5,000" in prompt_lower:
        return "HealthPlan 2026 covers up to $5,000 for dental procedures annually."
    if "grpc" in prompt_lower or "protocol" in prompt_lower:
        return "gRPC protocol is used for inter-service communication with 5ms SLA."
    if "latency" in prompt_lower:
        return "Hybrid FAISS Indexing reduced query latency from 85ms to 11.2ms."

    return "Based on the retrieved document context, relevant details were identified."


@pytest.fixture(scope="module")
def day15_faiss_store():
    """Build a shared FAISS store containing real PDF documents."""
    pdf_paths = [SAMPLE_PDF, AI_PDF, MARKET_PDF, MULTIPAGE_PDF]
    store, metadata = build_store(pdf_paths=pdf_paths)
    return store, metadata


@pytest.fixture(autouse=True)
def setup_day15_api_state():
    """Inject mock LLM into API state for endpoint integration testing."""
    state = get_state()
    state.client = mock_day15_llm
    state.model_name = "mock-gemini-day15"
    yield
    state.store = None
    state.sessions.clear()


# ------------------------------------------------------------
# 1. Grounded Response & Field Completeness Tests
# ------------------------------------------------------------

def test_backend_direct_context_processing():
    """Test receiving query + direct context from backend and returning all 4 mandatory fields."""
    direct_context = "The backend services use gRPC protocol for inter-service communication with a 5ms SLA."
    query = "What protocol is used for inter-service communication?"

    res = process_backend_query(
        client=mock_day15_llm,
        query=query,
        context=direct_context,
        source_document="tech_spec_2026.pdf",
        document_id="tech_spec_2026",
    )

    # Verify presence of mandatory structured fields
    assert "answer" in res
    assert "source_document" in res
    assert "document_id" in res
    assert "relevant_context" in res

    # Verify content correctness
    assert res["found"] is True
    assert "gRPC" in res["answer"]
    assert res["source_document"] == "tech_spec_2026.pdf"
    assert res["document_id"] == "tech_spec_2026"
    assert res["relevant_context"] == direct_context
    assert len(res["sources"]) == 1


def test_structured_response_field_presence_all_paths():
    """Verify answer, source_document, document_id, and relevant_context are present across all paths."""
    # Path A: Direct Context Success
    res_a = process_backend_query(
        client=mock_day15_llm,
        query="What is the dental limit?",
        context="HealthPlan 2026 covers up to $5,000 for dental procedures annually.",
        source_document="health.pdf",
        document_id="health",
    )
    for field in ("answer", "source_document", "document_id", "relevant_context"):
        assert field in res_a

    # Path B: Direct Context Missing / Fallback
    res_b = process_backend_query(
        client=mock_day15_llm,
        query="What is the alien policy?",
        context="",
        source_document="handbook.pdf",
        document_id="handbook",
    )
    for field in ("answer", "source_document", "document_id", "relevant_context"):
        assert field in res_b

    # Path C: Empty Query
    res_c = process_backend_query(
        client=mock_day15_llm,
        query="  ",
    )
    for field in ("answer", "source_document", "document_id", "relevant_context"):
        assert field in res_c


# ------------------------------------------------------------
# 2. Real Document Content Query Testing
# ------------------------------------------------------------

def test_real_document_multi_domain_queries(day15_faiss_store):
    """Test grounded answer generation against real documents across multiple domain topics."""
    store, metadata = day15_faiss_store

    domain_queries = [
        ("What was CyberPulse Systems revenue in Q3 2026?", "market_analysis_2026", "42.5 Million"),
        ("What microservices exist in OmniBrain?", "ai_architecture", "Document Parser Service"),
        ("What library is used for text extraction?", "sample", "pypdf"),
    ]

    for query, expected_doc_id, expected_keyword in domain_queries:
        res = process_backend_query(
            client=mock_day15_llm,
            query=query,
            store=store,
            min_score=0.01,
        )

        assert res["found"] is True
        assert res["document_id"] == expected_doc_id
        assert res["source_document"] == f"{expected_doc_id}.pdf"
        assert expected_keyword in res["answer"]
        assert len(res["relevant_context"]) > 0
        assert len(res["sources"]) > 0


# ------------------------------------------------------------
# 3. Fallback & Irrelevant Query Handling
# ------------------------------------------------------------

def test_irrelevant_query_fallback(day15_faiss_store):
    """Verify that irrelevant or out-of-domain queries receive an appropriate fallback response."""
    store, metadata = day15_faiss_store

    res = process_backend_query(
        client=mock_day15_llm,
        query="What is the distance from Earth to Mars in kilometers?",
        store=store,
        min_score=0.85,  # Strict threshold prevents irrelevant retrieval
    )

    assert res["found"] is False
    assert res["answer"] == NO_CONTEXT_MESSAGE
    assert res["source_document"] == "N/A"
    assert res["document_id"] == "N/A"
    assert res["relevant_context"] == ""
    assert res["sources"] == []
    assert res["retrieved_chunks"] == 0


def test_unsupported_query_incomplete_context():
    """Verify fallback response when context is provided but lacks facts to answer unsupported query."""
    incomplete_context = "Company perks include remote work flex hours and 15 days paid leave."
    query = "What is the policy for maternal leave?"

    res = process_backend_query(
        client=mock_day15_llm,
        query=query,
        context=incomplete_context,
        source_document="perks.pdf",
        document_id="perks",
    )

    assert res["found"] is False
    assert res["answer"] == NO_CONTEXT_MESSAGE
    assert res["source_document"] == "perks.pdf"
    assert res["document_id"] == "perks"
    assert res["relevant_context"] == incomplete_context
    assert res["sources"] == []


# ------------------------------------------------------------
# 4. Error and Failure Scenarios
# ------------------------------------------------------------

def test_error_empty_query():
    """Test error handling when backend passes empty or whitespace-only query."""
    res = process_backend_query(
        client=mock_day15_llm,
        query="   \t\n  ",
    )

    assert res["found"] is False
    assert "cannot be empty" in res["answer"].lower()
    assert res["source_document"] == "N/A"
    assert res["document_id"] == "N/A"
    assert res["relevant_context"] == ""


def test_error_llm_exception():
    """Test failure handling when LLM client raises an exception."""
    def failing_client(prompt: str) -> str:
        raise RuntimeError("LLM Service Outage")

    with pytest.raises(RuntimeError, match="LLM Service Outage"):
        process_backend_query(
            client=failing_client,
            query="What is the company revenue?",
            context="Revenue was $10M.",
        )


def test_error_uninitialized_store_and_context():
    """Test fallback when neither direct context nor store is provided."""
    res = process_backend_query(
        client=mock_day15_llm,
        query="What is the system latency?",
        context=None,
        store=None,
    )

    assert res["found"] is False
    assert res["answer"] == NO_CONTEXT_MESSAGE
    assert res["source_document"] == "N/A"
    assert res["document_id"] == "N/A"


# ------------------------------------------------------------
# 5. Pydantic Schema & REST API Integration
# ------------------------------------------------------------

def test_pydantic_schema_validation():
    """Validate BackendQueryInput and BackendQueryOutput schema serialization."""
    input_model = BackendQueryInput(
        query="What is the latency?",
        context="Latency is 11.2ms.",
        document_id="doc1",
        source_document="doc1.pdf",
    )

    res_raw = process_backend_query(
        client=mock_day15_llm,
        query=input_model.query,
        context=input_model.context,
        document_id=input_model.document_id,
        source_document=input_model.source_document,
    )

    sources = [SourceAttribution.from_dict(s) for s in res_raw.get("sources", [])]
    output_model = BackendQueryOutput(
        query=res_raw["query"],
        answer=res_raw["answer"],
        source_document=res_raw["source_document"],
        document_id=res_raw["document_id"],
        relevant_context=res_raw["relevant_context"],
        found=res_raw["found"],
        sources=sources,
        retrieved_chunks=res_raw["retrieved_chunks"],
        model=res_raw["model"],
    )

    assert output_model.found is True
    assert output_model.source_document == "doc1.pdf"
    assert output_model.document_id == "doc1"
    assert output_model.relevant_context == "Latency is 11.2ms."


def test_api_post_query_endpoint_backend_integration():
    """Verify REST API POST /query endpoint returns the consistent 4 mandatory fields."""
    with TestClient(app, raise_server_exceptions=True) as client:
        payload = {
            "query": "What protocol is used for inter-service communication?",
            "context": "The backend services use gRPC protocol for inter-service communication with a 5ms SLA.",
            "source_document": "tech_spec.pdf",
            "document_id": "tech_spec",
        }

        resp = client.post("/query", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert "answer" in data
        assert "source_document" in data
        assert "document_id" in data
        assert "relevant_context" in data
        assert data["found"] is True
        assert "gRPC" in data["answer"]
        assert data["source_document"] == "tech_spec.pdf"
        assert data["document_id"] == "tech_spec"
