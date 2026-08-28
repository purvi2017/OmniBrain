"""
test_day16_ai_response_generation.py
--------------------------------------
Day 16: AI Response Generation Test Suite.

Verifies:
1. Response generation from uploaded document content.
2. Connection of AI/RAG processing flow with user query requests.
3. Retrieval of relevant document context matching user questions.
4. Grounded answer generation using retrieved context.
5. Preparation of AI response payloads in backend-consumable formats.
6. Evaluation across sample document-based questions (Market Analysis, AI Spec, PDFs).
7. Object-oriented ResponseGenerator pipeline and generate_ai_response interface.
8. Serialization to Pydantic BackendQueryOutput models.
"""

import sys
from pathlib import Path
import pytest

# Ensure project root is on PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag_response_generator import ResponseGenerator, generate_ai_response
from rag_llm import build_store, NO_CONTEXT_MESSAGE
from schemas import BackendQueryInput, BackendQueryOutput, SourceAttribution

# Sample PDF documents
SAMPLE_PDF = str(BASE_DIR / "test_files" / "sample.pdf")
AI_PDF = str(BASE_DIR / "test_files" / "ai_architecture.pdf")
MARKET_PDF = str(BASE_DIR / "test_files" / "market_analysis_2026.pdf")
MULTIPAGE_PDF = str(BASE_DIR / "test_files" / "multipage_manual.pdf")


def mock_day16_llm(prompt: str) -> str:
    """Deterministic mock LLM client for Day 16 response generation tests."""
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
def day16_response_generator():
    """Create a ResponseGenerator instance with real documents indexed."""
    generator = ResponseGenerator(client=mock_day16_llm, min_score=0.01)
    generator.load_documents([SAMPLE_PDF, AI_PDF, MARKET_PDF, MULTIPAGE_PDF])
    return generator


# ------------------------------------------------------------
# 1. Document Context Retrieval & Grounded Answer Generation
# ------------------------------------------------------------

def test_response_generation_from_uploaded_document(day16_response_generator):
    """Test retrieving relevant context from uploaded document and generating grounded answer."""
    generator = day16_response_generator
    query = "What was CyberPulse Systems revenue in Q3 2026?"

    res = generator.generate_response(query=query)

    # 1. Verify connection of RAG processing flow with query request
    assert res["query"] == query
    assert res["found"] is True

    # 2. Verify answer generation using retrieved context
    assert "42.5 Million" in res["answer"]

    # 3. Verify backend-ready output structure
    assert res["source_document"] == "market_analysis_2026.pdf"
    assert res["document_id"] == "market_analysis_2026"
    assert len(res["relevant_context"]) > 0
    assert res["retrieved_chunks"] > 0
    assert len(res["sources"]) > 0


def test_sample_document_questions_eval(day16_response_generator):
    """Test AI response generation across multiple sample document-based questions."""
    generator = day16_response_generator

    sample_questions = [
        {
            "query": "What microservices exist in OmniBrain?",
            "expected_doc": "ai_architecture",
            "expected_keyword": "Document Parser Service",
        },
        {
            "query": "What library is used for text extraction?",
            "expected_doc": "sample",
            "expected_keyword": "pypdf",
        },
        {
            "query": "What was CyberPulse Systems revenue in Q3 2026?",
            "expected_doc": "market_analysis_2026",
            "expected_keyword": "42.5 Million",
        },
    ]

    for sq in sample_questions:
        res = generator.generate_response(query=sq["query"])

        assert res["found"] is True, f"Failed for question: {sq['query']}"
        assert res["document_id"] == sq["expected_doc"]
        assert sq["expected_keyword"] in res["answer"]
        assert len(res["relevant_context"]) > 0
        assert len(res["sources"]) > 0


# ------------------------------------------------------------
# 2. Backend Format & Schema Validation
# ------------------------------------------------------------

def test_backend_response_format_structure(day16_response_generator):
    """Verify response format contains all required keys for backend consumption."""
    generator = day16_response_generator
    res = generator.generate_response(query="What was CyberPulse Systems revenue in Q3 2026?")

    required_keys = {
        "query",
        "answer",
        "source_document",
        "document_id",
        "relevant_context",
        "found",
        "sources",
        "retrieved_chunks",
        "model",
    }
    assert required_keys.issubset(res.keys())


def test_typed_backend_response_pydantic(day16_response_generator):
    """Verify generate_typed_response returns a validated BackendQueryOutput model."""
    generator = day16_response_generator
    input_payload = BackendQueryInput(
        query="What library is used for text extraction?",
        document_id="sample",
        min_score=0.01,
    )

    output_model = generator.generate_typed_response(input_payload)

    assert isinstance(output_model, BackendQueryOutput)
    assert output_model.found is True
    assert output_model.document_id == "sample"
    assert "pypdf" in output_model.answer
    assert isinstance(output_model.sources[0], SourceAttribution)


# ------------------------------------------------------------
# 3. Direct Context & Standalone Function Interface
# ------------------------------------------------------------

def test_generate_ai_response_standalone_function():
    """Test generate_ai_response standalone function interface with direct context."""
    direct_context = "HealthPlan 2026 covers up to $5,000 for dental procedures annually."
    query = "What is the annual dental coverage limit?"

    res = generate_ai_response(
        client=mock_day16_llm,
        query=query,
        context=direct_context,
        source_document="health_policy.pdf",
        document_id="health_policy",
    )

    assert res["found"] is True
    assert "$5,000" in res["answer"]
    assert res["source_document"] == "health_policy.pdf"
    assert res["document_id"] == "health_policy"
    assert res["relevant_context"] == direct_context


# ------------------------------------------------------------
# 4. Fallback & Error Scenarios
# ------------------------------------------------------------

def test_response_generation_unsupported_question(day16_response_generator):
    """Verify fallback answer for unsupported / out-of-domain document question."""
    generator = day16_response_generator
    query = "What is the orbital speed of Jupiter?"

    res = generator.generate_response(query=query, min_score=0.85)

    assert res["found"] is False
    assert res["answer"] == NO_CONTEXT_MESSAGE
    assert res["source_document"] == "N/A"
    assert res["document_id"] == "N/A"
    assert res["relevant_context"] == ""
    assert res["retrieved_chunks"] == 0


def test_response_generation_empty_query(day16_response_generator):
    """Verify error handling when query request is empty."""
    generator = day16_response_generator
    res = generator.generate_response(query="   ")

    assert res["found"] is False
    assert "cannot be empty" in res["answer"].lower()
    assert res["source_document"] == "N/A"
    assert res["document_id"] == "N/A"
