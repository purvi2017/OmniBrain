# AI Module

## Overview

The AI module is an enterprise-grade Retrieval-Augmented Generation (RAG) system that processes multi-page documents (such as PDFs), indexes their contents into a FAISS vector database, and generates grounded, source-attributed answers using Large Language Models (Gemini API via `google-genai`).

---

## RAG + LLM Architecture & Expected Flow

```text
User Query
    ↓
Query Embedding (all-MiniLM-L6-v2)
    ↓
Multi-Document FAISS Search (IndexFlatIP Cosine Similarity)
    ↓
Adaptive Threshold & Delta Filtering (min_score >= 0.35, max_drop <= 0.25)
    ↓
Page-Level Context Assembly (Numbered Chunks + Doc IDs + Page Numbers)
    ↓
Gemini LLM (Strict Grounding & Anti-Hallucination Prompt)
    ↓
Grounded Answer Generation
    ↓
Structured Response (Answer + Page-Level Source Citations + Confidence Scores)
```

---

## Day 10: FastAPI REST Microservice

- **Deployable REST API**: `api.py` exposes the full RAG system as a production-ready HTTP microservice built with FastAPI and Uvicorn.
- **Stateless & Scalable Endpoints**:
  - `GET /health` — Liveness check, index statistics, active sessions.
  - `POST /ingest` — Upload and index one or multiple PDF documents into FAISS.
  - `POST /query` — Single-turn grounded RAG question answering.
  - `POST /chat/{session_id}` — Stateful multi-turn conversational chat with history.
  - `GET /sessions` — List active sessions and message counts.
  - `GET /sessions/{session_id}` — Retrieve full conversation history.
  - `DELETE /sessions/{session_id}` — Clear a session's history.
- **Pydantic Validation**: Strict request/response schema validation for all endpoints.

---

## Day 9: Conversational RAG with Chat History & Session Management

- **Multi-Turn Conversation**: `rag_chat.py` wraps the Day 8 pipeline with a `ChatSession` that tracks Q&A history across turns, enabling natural follow-up questions like *"What else does it say about that?"*
- **History-Aware Prompting**: Each LLM call receives a sliding window of prior turns (configurable `--history-window`) injected above the retrieved document context, so the model can resolve pronoun references without hallucinating.
- **`ChatMessage` Data Model**: Typed dataclass (`role`, `content`, `timestamp`, `sources`) for structured message representation.
- **`ChatSession` Manager**:
  - `add_message()` — append user or assistant turns with strict role validation.
  - `get_window()` — return the last N complete Q&A turn pairs for prompt injection.
  - `get_history_prompt()` — format the window as a human-readable block.
  - `clear()` — reset history without losing the session object.
  - `turn_count()` — count complete Q&A pairs.
- **Session Persistence**: `save()` / `load()` serialise full conversation history to JSON, enabling sessions to resume across runs.
- **`ConversationalRAGPipeline`**: Object-oriented interface integrating `ChatSession` with `RAGLLMPipeline`.
  - `chat()` — single-method entry point: retrieve → inject history → generate answer → record turn.
  - `reset()` — clear session history.
  - `save_session()` / `load_session()` — persist and restore sessions.
- **Interactive REPL CLI**: Built-in commands: `/reset`, `/save`, `/history`, `/quit`.
- **Strict Grounding Preserved**: History is injected for reference resolution only; all facts must still come from the retrieved document context.

---

## Day 8 & Document Chunking Engine Improvements

- **Intelligent Sentence Boundary Tokenization**:
  - Sentence splitter accurately protects honorifics/titles (`Dr.`, `Mr.`, `Mrs.`, `Prof.`), Latin abbreviations (`e.g.`, `i.e.`, `et al.`, `vs.`, `cf.`), citations & technical terms (`Fig.`, `Tab.`, `Vol.`, `Inc.`, `Ltd.`), decimal numbers (`3.14`, `$14.50`), URLs, and emails.
  - Properly respects list items (`1.`, `2.`, `•`, `-`) and paragraph breaks without premature sentence fragmentation.
- **Word-Safe & Sentence-Aware Overlap**:
  - Overlap slices strictly respect word and sentence boundaries to avoid slicing words mid-token, maintaining full semantic fidelity across chunk windows.
- **Graceful Edge Case Handling**:
  - Handles empty, whitespace-only, `None`, very short texts (single words / short sentences), and oversized unbroken paragraphs cleanly.
- **Pre-Embedding Chunk Verification**:
  - `verify_chunks()` and `validate_and_filter_chunks()` audit and sanitize chunks prior to embedding, verifying bounds, lengths, and non-emptiness.
- **Multi-Document Ingestion**: Seamless ingestion and simultaneous indexing of multiple PDFs into a unified FAISS index with distinct `document_id`, `filename`, and `page` metadata.
- **Adaptive Similarity Thresholding**:
  - **Calibrated Baseline**: Default `min_score = 0.35` tuned for `all-MiniLM-L6-v2` cosine similarity.
  - **Relative Score Drop Filtering**: `max_score_drop = 0.25` prevents low-relevance tail noise chunks from polluting LLM context even when larger `top_k` is requested.
- **Page-Aware Extraction & Chunking**: `extract_pages_from_pdf()` and `chunk_pages()` retain page boundaries throughout chunking and indexing.
- **Enhanced Source Citations**: Each retrieved source includes `document_id`, `filename`, `page`, `chunk_id`, `score`, `confidence` (`HIGH` / `MEDIUM` / `LOW`), and `text_preview`.
- **Targeted Document Filtering**: Ability to constrain queries to a specific document via `--document-id`.
- **Deterministic Out-of-Domain Rejection**: Returns a clean structured fallback when no relevant context passes the threshold.

---

## Project Structure

```text
ai-module/
│
├── parsers/
│   ├── __init__.py
│   ├── pdf_extractor.py          # PDF text & page extraction (pypdf/PyMuPDF)
│   ├── image_extractor.py        # PDF image extraction
│   └── chunker.py                # Page-aware & overlapping text chunker
│
├── embeddings/
│   ├── __init__.py
│   └── embedder.py               # Sentence-Transformers embedder (all-MiniLM-L6-v2)
│
├── vector_db/
│   ├── __init__.py
│   ├── faiss_store.py            # FAISS vector store with page tracking & persistence
│   └── qdrant_store.py           # Optional Qdrant store
│
├── tests/
│   ├── __init__.py
│   ├── test_extraction.py        # PDF extraction tests
│   ├── test_chunking.py          # Chunking boundary tests
│   ├── test_embeddings.py        # Vector embedding tests
│   ├── test_faiss_storage.py     # FAISS storage & persistence tests
│   ├── test_similarity_search.py # FAISS semantic search tests
│   ├── test_chunk_optimization.py# Chunk size optimization benchmarks
│   ├── test_rag_llm.py           # RAG + LLM pipeline tests
│   ├── test_rag_retrieval_accuracy.py # Day 8 Multi-Doc & retrieval accuracy test suite
│   ├── test_rag_chat.py          # Day 9 Conversational RAG & session management tests
│   └── test_api.py               # Day 10 FastAPI REST API test suite
│
├── test_files/
│   ├── sample.pdf                # Verification test document 1
│   └── ai_architecture.pdf       # Multi-page test document 2
│
├── tools/
│   └── render_screenshot.py      # Terminal output screenshot renderer
│
├── process_document.py           # Ingestion & index pipeline CLI
├── rag_retriever.py              # Enhanced semantic retrieval pipeline
├── rag_llm.py                    # Multi-document RAG + LLM pipeline
├── rag_chat.py                   # Day 9: Conversational RAG with chat history
├── api.py                        # Day 10: FastAPI REST API microservice
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

---

## Installation & Setup

1. **Activate Virtual Environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure Gemini API Key**:
   ```powershell
   $env:GEMINI_API_KEY = "your_gemini_api_key_here"
   ```

---

## Usage

### 1. REST API Server (Day 10)

Start the Uvicorn REST server:

```powershell
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Access Interactive API Documentation (Swagger UI) at `http://localhost:8000/docs`.

Example API requests:

```powershell
# Health check
curl http://localhost:8000/health

# Ingest PDFs
curl -X POST http://localhost:8000/ingest -F "files=@test_files/sample.pdf" -F "files=@test_files/ai_architecture.pdf"

# Single-turn query
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"query\": \"What are the core microservices?\"}"

# Conversational chat
curl -X POST http://localhost:8000/chat/session1 -H "Content-Type: application/json" -d "{\"message\": \"What is FAISS?\"}"
```

### 2. Conversational RAG CLI (Day 9)

Start an interactive multi-turn chat session with one or more PDFs:

```powershell
# Interactive conversational session with two documents
python rag_chat.py test_files/sample.pdf test_files/ai_architecture.pdf

# Custom history window and score threshold
python rag_chat.py test_files/sample.pdf --history-window 5 --min-score 0.35

# Auto-save session on exit and resume it next time
python rag_chat.py test_files/sample.pdf --save-session chat_sessions/my_session.json
python rag_chat.py test_files/sample.pdf --load-session chat_sessions/my_session.json
```

In-session commands:

| Command | Action |
|---------|--------|
| `/reset` | Clear conversation history |
| `/save` | Save current session to JSON |
| `/history` | Show prior turns in terminal |
| `/quit` | Exit and (optionally) save |

### 2. Conversational Python API

```python
from rag_chat import ConversationalRAGPipeline

pipeline = ConversationalRAGPipeline(
    model_name="gemini-2.5-flash",
    top_k=3,
    min_score=0.35,
    history_window=6,
)

pipeline.load_documents([
    "test_files/sample.pdf",
    "test_files/ai_architecture.pdf"
])

# Turn 1
r1 = pipeline.chat("What are the core microservices in the OmniBrain AI architecture?")
print(r1["answer"])

# Turn 2 — follow-up resolved using history
r2 = pipeline.chat("Which of those communicates directly with Gemini?")
print(r2["answer"])   # Knows 'those' refers to the microservices from Turn 1

# Persist session
pipeline.save_session("chat_sessions/session1.json")
```

### 3. Multi-Document RAG CLI (Day 8)

Ask questions across single or multiple PDF documents:

```powershell
# Multi-document query across sample.pdf and ai_architecture.pdf
python rag_llm.py test_files/sample.pdf test_files/ai_architecture.pdf --query "How does the microservices architecture communicate?"

# Constrain query to a specific document
python rag_llm.py test_files/sample.pdf test_files/ai_architecture.pdf --query "What library is used for text extraction?" --document-id sample

# Custom retrieval tuning parameters
python rag_llm.py doc1.pdf doc2.pdf --query "What are the evaluation metrics?" --top-k 3 --min-score 0.35 --max-drop 0.25 --model gemini-2.5-flash
```

### 2. Programmatic Python API

```python
from rag_llm import RAGLLMPipeline

# Initialize Multi-Doc RAG pipeline
pipeline = RAGLLMPipeline(
    model_name="gemini-2.5-flash",
    top_k=3,
    min_score=0.35,
    max_score_drop=0.25,
)

# Index multiple PDF documents
pipeline.load_documents([
    "test_files/sample.pdf",
    "test_files/ai_architecture.pdf"
])

# Query across all indexed documents
response = pipeline.query("What are the three evaluation metrics for retrieval precision?")

print("Answer:", response["answer"])
print("Found Context:", response["found"])
for src in response["sources"]:
    print(f"- {src['filename']} (Page {src['page']}) | Score: {src['score']} ({src['confidence']})")
```

---

## Structured Response Format

Every query produces a structured dictionary:

```json
{
  "query": "What are the core microservices in the OmniBrain AI architecture?",
  "answer": "According to ai_architecture.pdf (Page 1), the core microservices include the Document Parser Service, the Embedding Generation Service, the Vector Storage Engine, and the LLM Orchestrator.",
  "found": true,
  "sources": [
    {
      "source": "ai_architecture.pdf",
      "filename": "ai_architecture.pdf",
      "document_id": "ai_architecture",
      "chunk_id": 2,
      "page": 1,
      "score": 0.8446,
      "confidence": "HIGH",
      "text_preview": "AI System Architecture and Microservices Overview: The OmniBrain AI system is organized into decoupled microservices..."
    }
  ],
  "retrieved_chunks": 1,
  "model": "gemini-2.5-flash"
}
```

### Out-of-Domain Query Handling

When a query has no relevant facts in the indexed documents:

```json
{
  "query": "What is the average surface temperature of Venus?",
  "answer": "I could not find relevant information in the uploaded documents to answer this question.",
  "found": false,
  "sources": [],
  "retrieved_chunks": 0,
  "model": "gemini-2.5-flash"
}
```

---

## Running Verification Tests

Run the complete automated pytest suite (38 tests covering chunking, embeddings, extraction, FAISS storage, RAG retrieval, accuracy, and conversational RAG):

```powershell
pytest tests/ -v
```

Run only the Day 9 conversational RAG test suite:

```powershell
pytest tests/test_rag_chat.py -v
```

Run the Chunk Size and Overlap Optimization Multi-PDF benchmark:

```powershell
python tests/test_chunk_optimization.py
```

Run the Multi-Doc Retrieval Accuracy test suite:

```powershell
python tests/test_rag_retrieval_accuracy.py
```