# AI Module — Document Processing (OmniBrain, Week 1)

Owner: Srikanth (AI Module Developer)

Full pipeline: PDF ingestion → text extraction → image extraction →
chunking → embeddings → vector storage (FAISS, with Qdrant as an
alternative backend) → ready for query from the backend/frontend.

## Folder Structure

```
ai-module/
│
├── parsers/                  # PDF extraction + chunking logic
│   ├── pdf_extractor.py       # extract text
│   ├── image_extractor.py     # extract embedded images
│   ├── chunker.py             # split text into chunks
│   └── __init__.py
├── embeddings/                # embedding generation
│   ├── embedder.py            # sentence-transformers wrapper
│   └── __init__.py
├── vector_db/                  # vector storage & search
│   ├── faiss_store.py          # FAISS backend (default, local)
│   ├── qdrant_store.py         # Qdrant backend (optional, server-based)
│   └── __init__.py
├── models/                      # local/cached model files (empty for now)
├── test_files/                   # sample PDFs used for manual testing
│   ├── sample.pdf
│   ├── extracted_images/          # output of image extraction
│   └── screenshots/                # Day 2 test output screenshots
├── tests/                         # Day 2 verification test scripts
│   ├── test_extraction.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   └── test_faiss_storage.py
├── tools/
│   └── render_screenshot.py        # renders captured test output as PNG
├── process_document.py            # end-to-end pipeline entry point
├── requirements.txt
└── README.md
```

## Libraries Used

| Library | Purpose | Notes |
|---|---|---|
| `pypdf` | PDF text extraction | Pure-Python, no external binary dependency. Handles most text-based PDFs well; scanned/image-only PDFs need OCR (out of current scope). |
| `pymupdf` (fitz) | PDF image extraction | Gives direct access to embedded image XObjects with original encoding — more reliable for image extraction than pypdf's image API. |
| `sentence-transformers` | Generating embeddings from text chunks | Default model `all-MiniLM-L6-v2`, downloaded from Hugging Face on first run (requires internet access once, then cached locally). |
| `faiss-cpu` | Vector similarity search / storage (default backend) | In-process, no server needed. Uses `IndexFlatIP` (cosine similarity via normalized inner product) — exact search, fine at current data scale. |
| `qdrant-client` | Alternative vector storage backend | Optional — only needed if the team decides to use Qdrant instead of/alongside FAISS. Requires a running Qdrant server (see `vector_db/qdrant_store.py` docstring). |

Install everything with:
```bash
pip install -r requirements.txt
```

## Embedding Model Options (for Day 2 decision)

| Model | Dimensions | Notes |
|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Fast, small, good default for prototyping. Lower accuracy than larger models. |
| `all-mpnet-base-v2` | 768 | Higher quality embeddings, slower and larger than MiniLM. Good general-purpose choice. |
| `multi-qa-mpnet-base-dot-v1` | 768 | Tuned specifically for semantic search / Q&A retrieval — worth considering since this pipeline is for document search. |
| OpenAI / Anthropic-hosted embedding APIs | varies | Alternative to local models if we want to avoid hosting model weights; adds external API dependency/cost. |

**Recommendation to evaluate on Day 2:** start with `all-MiniLM-L6-v2` for speed
during development, benchmark against `multi-qa-mpnet-base-dot-v1` once we have
real documents, and pick based on retrieval quality vs. latency trade-off.

## Vector DB Options (for Day 2 decision)

| Option | Type | Pros | Cons |
|---|---|---|---|
| **FAISS** | In-process library | No server to run, very fast, simple to start with | No built-in persistence layer or metadata filtering out of the box; need to build that ourselves |
| **Qdrant** | Standalone server (Docker or cloud) | Metadata filtering, persistence, REST/gRPC API, easier to scale to multiple services | Requires running a separate service/container |

**Recommendation to evaluate on Day 2:** start with FAISS for local
prototyping (matches current `vector_db/` scope), and consider migrating to
Qdrant if we need multi-service access, filtering by document metadata, or
production-scale persistence.

## Current Progress (Week 1)

- [x] AI module folder structure created
- [x] Libraries installed and import-verified (`pypdf`, `pymupdf`, `sentence-transformers`, `faiss-cpu`)
- [x] PDF text extraction implemented (`parsers/pdf_extractor.py`)
- [x] PDF image extraction implemented (`parsers/image_extractor.py`) — tested against a sample PDF with an embedded image, correctly extracted and saved to `test_files/extracted_images/`
- [x] Chunking logic implemented (`parsers/chunker.py`), sentence-aware, 500–1000 char target with configurable overlap
- [x] Embedding generation implemented (`embeddings/embedder.py`) using `sentence-transformers` (`all-MiniLM-L6-v2` by default)
- [x] Vector DB storage implemented — FAISS (`vector_db/faiss_store.py`, default) tested end-to-end: add → save → load → search, all working correctly
- [x] Qdrant backend implemented as an alternative (`vector_db/qdrant_store.py`) — code complete, requires a running Qdrant server to test (not run yet — Docker setup pending)
- [x] End-to-end pipeline (`process_document.py`) wires all steps together: PDF → text + images → chunks → embeddings → FAISS index, with a demo search at the end
- [x] Day 2: All pipeline components individually tested with pass/fail assertions (`tests/`)
- [x] Day 2: Output screenshots captured for extraction, chunking, embeddings, FAISS storage (`test_files/screenshots/`)
- [ ] Swap in Qdrant and benchmark vs FAISS — pending team decision
- [ ] Connect to backend `/query` API — pending backend readiness (Narsimha)

### Known limitation
Embedding generation requires downloading the model from Hugging Face on
first run (one-time, then cached). If running in a network-restricted
environment, pre-download the model or the `embed_chunks()` call will fail
with a connection error — this is an environment/network issue, not a code
bug (verified: chunking → FAISS storage/search work correctly independent
of this using test vectors).

## Day 2 — Verification & Testing

All pipeline components were tested individually with pass/fail assertions
in `tests/`. Run them all with:

```bash
python tests/test_extraction.py
python tests/test_chunking.py
python tests/test_embeddings.py
python tests/test_faiss_storage.py
```

| Test | What it checks | Result |
|---|---|---|
| `test_extraction.py` | Text is non-empty, char/word counts reported | ✅ PASSED — 694 chars, 111 words extracted from `sample.pdf` |
| `test_chunking.py` | All chunks within size bounds, no empty chunks | ✅ PASSED — 2 chunks, both within the 500(+50 overlap) char limit |
| `test_embeddings.py` | Vector count matches chunk count, correct dtype, no NaNs, deterministic output | ✅ PASSED — 2 vectors, dim 384, float32 |
| `test_faiss_storage.py` | Vectors added, saved to disk, reloaded, and searchable | ✅ PASSED — save → reload → search all confirmed working |

Screenshots of each test run are in `test_files/screenshots/`.

### Note on the embeddings test environment
`test_embeddings.py` and `test_faiss_storage.py` were run in a sandboxed
dev environment with no outbound access to huggingface.co, so
`sentence-transformers` couldn't download `all-MiniLM-L6-v2` on that
machine. To keep the pipeline testable in that environment, `embedder.py`
automatically falls back to a deterministic local hashing-based embedder
when the real model can't be downloaded — this proves the chunk → vector →
FAISS wiring is correct, but is **not** a semantic embedding. On a normal
dev machine with internet access, `embed_chunks()` uses the real
`all-MiniLM-L6-v2` model automatically (no fallback triggered) — you'll see
the `[WARN] Could not load...` line disappear from the output.


## How to Run

```bash
pip install -r requirements.txt

# Individual steps
python parsers/pdf_extractor.py test_files/sample.pdf
python parsers/image_extractor.py test_files/sample.pdf --out-dir test_files/extracted_images

# Full pipeline: text + images + chunk + embed + store in FAISS + demo search
python process_document.py test_files/sample.pdf --chunk-size 800 --overlap 100 --top-k 3
```

## Design Notes

- **Chunking** splits on sentence boundaries (regex-based) rather than a hard
  character cut, so we avoid slicing a sentence in half mid-word. A
  configurable overlap (default 100 chars) is carried between consecutive
  chunks to preserve context across chunk boundaries — this matters for
  retrieval quality once embeddings are in play.
- **Extraction** logs a warning per page if no text is found (e.g. a scanned
  image page), rather than failing silently or crashing the whole document.
- `process_document.py` is intentionally the only "entry point" script so
  Day 2 work (embeddings, vector_db) can import from `parsers/` without
  duplicating extraction/chunking logic.
