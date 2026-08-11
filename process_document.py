"""
process_document.py
--------------------
Full AI-module pipeline (Week 1 deliverable):
Reads a PDF -> extracts text -> extracts images -> chunks text ->
generates embeddings -> stores embeddings in FAISS -> ready for query.

Usage:
    python process_document.py test_files/sample.pdf
    python process_document.py test_files/sample.pdf --chunk-size 800 --overlap 100 --top-k 3
"""

import argparse
import os
import sys

from parsers.pdf_extractor import extract_text_from_pdf
from parsers.image_extractor import extract_images_from_pdf
from parsers.chunker import chunk_text
from embeddings.embedder import embed_chunks
from vector_db.faiss_store import FaissStore


def process_document(pdf_path: str, chunk_size: int = 800, overlap: int = 100,
                      image_out_dir: str = "test_files/extracted_images",
                      index_out_prefix: str = "vector_db/index_data"):
    pdf_name = os.path.basename(pdf_path)

    # 1. Extract text
    print(f"[STEP 1] Extracting text from: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("[ERROR] No text extracted. Cannot proceed.")
        return None
    print(f"[STEP 1] Done. Extracted {len(text)} characters.\n")

    # 2. Extract images
    print(f"[STEP 2] Extracting images...")
    image_paths = extract_images_from_pdf(pdf_path, image_out_dir)
    print(f"[STEP 2] Done. Extracted {len(image_paths)} image(s) to '{image_out_dir}'.\n")

    # 3. Chunk text
    print(f"[STEP 3] Chunking text (chunk_size={chunk_size}, overlap={overlap})...")
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    print(f"[STEP 3] Done. Created {len(chunks)} chunks.\n")

    # 4. Generate embeddings
    print(f"[STEP 4] Generating embeddings for {len(chunks)} chunks...")
    vectors = embed_chunks(chunks)
    print(f"[STEP 4] Done. Embeddings shape: {vectors.shape}\n")

    # 5. Store in FAISS
    print(f"[STEP 5] Storing embeddings in FAISS index...")
    store = FaissStore(dim=vectors.shape[1])
    store.add(vectors, chunks, source=pdf_name)
    store.save(index_out_prefix)
    print(f"[STEP 5] Done. Index saved to '{index_out_prefix}.index'.\n")

    return {
        "text_length": len(text),
        "images": image_paths,
        "chunks": chunks,
        "vectors": vectors,
        "store": store,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: PDF -> text+images -> chunks -> embeddings -> FAISS."
    )
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3,
                         help="Number of results to show in the demo search at the end")
    args = parser.parse_args()

    result = process_document(args.pdf_path, args.chunk_size, args.overlap)
    if result is None:
        sys.exit(1)

    # Demo: search the index using the first chunk as a query, just to prove
    # the retrieval pipeline works end-to-end.
    print("[DEMO] Running a sample search using the first chunk as the query...")
    query_vector = result["vectors"][0]
    hits = result["store"].search(query_vector, top_k=args.top_k)
    for i, hit in enumerate(hits, start=1):
        preview = hit["text"][:100].replace("\n", " ")
        print(f"  #{i} score={hit['score']:.4f} source={hit['source']} text={preview}...")

    print(f"\n[DONE] Pipeline complete: {len(result['chunks'])} chunks embedded and stored, "
          f"{len(result['images'])} images extracted.")


if __name__ == "__main__":
    main()
