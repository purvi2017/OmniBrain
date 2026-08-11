"""
pdf_extractor.py
----------------
Reads a PDF file and extracts its raw text content page by page
using the `pypdf` library.

Usage (as a script):
    python pdf_extractor.py path/to/sample.pdf

Usage (as a module):
    from parsers.pdf_extractor import extract_text_from_pdf
    text = extract_text_from_pdf("path/to/sample.pdf")
"""

import sys
import os
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A single string containing text from all pages,
        separated by newlines. Returns an empty string if
        no text could be extracted (e.g. scanned/image-only PDF).

    Raises:
        FileNotFoundError: If the given path does not exist.
        ValueError: If the file is not a valid PDF.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        raise ValueError(f"Could not read PDF '{pdf_path}': {e}")

    all_text = []
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            all_text.append(page_text)
        else:
            print(f"[WARN] No extractable text on page {page_num} "
                  f"(may be an image/scanned page).")

    return "\n".join(all_text)


def main():
    if len(sys.argv) != 2:
        print("Usage: python pdf_extractor.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        text = extract_text_from_pdf(pdf_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not text:
        print("[INFO] No text extracted from this PDF.")
        return

    print("----- Extracted Text -----")
    print(text)
    print("---------------------------")
    print(f"[INFO] Total characters extracted: {len(text)}")


if __name__ == "__main__":
    main()
