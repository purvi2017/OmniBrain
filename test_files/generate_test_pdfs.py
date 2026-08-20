"""
generate_test_pdfs.py
---------------------
Generates diverse test PDF documents using PyMuPDF (fitz) to thoroughly test
document chunking, boundary preservation, edge cases, and multi-PDF pipelines.
"""

import os
import pymupdf as fitz


def create_complex_formatting_pdf(output_path: str):
    """Creates a PDF with complex formatting: abbreviations, decimal numbers, lists, quotes."""
    doc = fitz.open()
    page = doc.new_page()

    text = """Enterprise AI Architecture Specification (v2.4.1)

1. Executive Summary
Dr. Evelyn Vance and Prof. Marcus Cole published findings regarding distributed retrieval engines (e.g., FAISS, Qdrant, and Milvus).
Their research demonstrated a 99.85% precision rate with an operational cost of $14.50 per 1M processed queries.
As noted by Vance et al., "Sentence-aware boundary preservation prevents severe context fragmentation."

2. Key Microservices & Components
The system consists of the following decoupled microservices:
• Ingestion Service: Parses incoming PDF documents (e.g. research papers, technical specs, user manuals).
• Normalization Layer: Cleans text formatting, protects abbreviations (e.g., Dr., vs., etc.), and handles unicode.
• Chunking Engine: Segments text into 500-1000 character windows with 100 character sentence-safe overlap.
• Vector Embedding Engine: Transforms chunks into 384-dimensional dense vectors using all-MiniLM-L6-v2.
• Similarity Retrieval Store: Performs cosine distance indexing via FAISS IndexFlatIP.

3. Performance Benchmarks
Figure 1.1 illustrates latency curves under high concurrency. Average query latency was 12.4ms (vs. 45.2ms in legacy architectures).
System throughput peaked at 4,500 requests/sec with zero packet loss across all tested AWS regions (i.e. us-east-1, eu-west-1).
"""
    # Insert text with nice margin
    rect = fitz.Rect(50, 50, 550, 750)
    page.insert_textbox(rect, text, fontsize=11, fontname="helv", align=0)
    doc.save(output_path)
    doc.close()
    print(f"[CREATED] {output_path}")


def create_multipage_manual_pdf(output_path: str):
    """Creates a 3-page technical manual with varied content and structure."""
    doc = fitz.open()

    # Page 1: Introduction & System Setup
    page1 = doc.new_page()
    text_p1 = """Cloud-Native RAG Deployment Manual - Page 1

Chapter 1: Getting Started and Prerequisites
Welcome to the Cloud-Native Retrieval-Augmented Generation deployment guide.
Before beginning installation, ensure your environment meets the minimum hardware and software prerequisites:
- Python 3.10 or higher installed with pip and virtualenv support.
- At least 8GB of system RAM (16GB recommended for batch embedding workloads).
- CUDA-compatible GPU with minimum 4GB VRAM (optional, CPU execution supported via faiss-cpu).

Installation Steps:
1. Clone the repository from the centralized git hosting platform.
2. Initialize and activate an isolated Python virtual environment.
3. Install required runtime dependencies using the requirements lockfile.
4. Verify environment readiness by running the automated self-test diagnostic suite.
"""
    page1.insert_textbox(fitz.Rect(50, 50, 550, 750), text_p1, fontsize=11, fontname="helv")

    # Page 2: Chunking Strategy & Ingestion Details
    page2 = doc.new_page()
    text_p2 = """Cloud-Native RAG Deployment Manual - Page 2

Chapter 2: Text Ingestion and Sentence Chunking Architecture
Document chunking represents the cornerstone of effective semantic retrieval.
When documents are poorly partitioned, LLMs suffer from incomplete context or broken semantic phrases.

Key Rules for Chunk Configuration:
- Target Chunk Size: Recommended between 500 and 1000 characters. Chunks smaller than 300 characters lack semantic context, while chunks larger than 1500 characters dilute vector precision.
- Overlap Ratio: Maintain 10-15% character overlap between adjacent chunks to guarantee sentence continuity.
- Abbreviation Safety: Ensure abbreviations such as Dr., Prof., e.g., and i.e. do not prematurely trigger sentence splits.
- Word Boundary Preservation: Never slice words mid-token during character fallback.
"""
    page2.insert_textbox(fitz.Rect(50, 50, 550, 750), text_p2, fontsize=11, fontname="helv")

    # Page 3: Maintenance & Troubleshooting
    page3 = doc.new_page()
    text_p3 = """Cloud-Native RAG Deployment Manual - Page 3

Chapter 3: Maintenance, Monitoring, and Troubleshooting
Routine maintenance ensures vector store indexes remain optimized for low-latency queries.

Monitoring Recommendations:
- Track average search latency per query (target: < 20ms).
- Monitor index memory footprint (FAISS FlatIP consumes ~1.5KB per indexed 384-d vector).
- Audit chunk verification logs for empty or oversized chunks prior to embedding ingestion.

Troubleshooting Common Issues:
If retrieval scores drop below the 0.35 confidence threshold, re-evaluate chunk sizes and document parsing quality.
Ensure that scanned or image-only documents undergo Optical Character Recognition (OCR) before ingestion.
"""
    page3.insert_textbox(fitz.Rect(50, 50, 550, 750), text_p3, fontsize=11, fontname="helv")

    doc.save(output_path)
    doc.close()
    print(f"[CREATED] {output_path}")


def create_short_document_pdf(output_path: str):
    """Creates a very short document with only one brief sentence."""
    doc = fitz.open()
    page = doc.new_page()
    text = "Quick Summary: System status is operational."
    page.insert_textbox(fitz.Rect(50, 50, 550, 750), text, fontsize=12, fontname="helv")
    doc.save(output_path)
    doc.close()
    print(f"[CREATED] {output_path}")


def create_empty_or_scanned_pdf(output_path: str):
    """Creates a PDF with an empty page (zero extractable text)."""
    doc = fitz.open()
    page = doc.new_page()
    # Draw a simple rectangle simulating an image or blank page without text
    rect = fitz.Rect(100, 100, 400, 400)
    page.draw_rect(rect, color=(0.8, 0.8, 0.8), fill=(0.95, 0.95, 0.95))
    doc.save(output_path)
    doc.close()
    print(f"[CREATED] {output_path}")


def main():
    target_dir = os.path.dirname(os.path.abspath(__file__))
    create_complex_formatting_pdf(os.path.join(target_dir, "complex_formatting.pdf"))
    create_multipage_manual_pdf(os.path.join(target_dir, "multipage_manual.pdf"))
    create_short_document_pdf(os.path.join(target_dir, "short_document.pdf"))
    create_empty_or_scanned_pdf(os.path.join(target_dir, "empty_or_scanned.pdf"))
    print("[ALL PDFS GENERATED SUCCESSFULLY]")


if __name__ == "__main__":
    main()
