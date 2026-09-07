"""
image_extractor.py
-------------------
Extracts embedded images from a PDF file using PyMuPDF (fitz).

PyMuPDF is used here instead of pypdf because it gives direct access
to each page's embedded image XObjects with their original encoding,
which is more reliable for image extraction than pypdf's image API.

Usage (as a script):
    python image_extractor.py path/to/sample.pdf --out-dir extracted_images

Usage (as a module):
    from parsers.image_extractor import extract_images_from_pdf
    saved_paths = extract_images_from_pdf("sample.pdf", "extracted_images")
"""

import os
import sys
import argparse
from typing import List

import pymupdf  # PyMuPDF (formerly imported as `fitz`)


def extract_images_from_pdf(pdf_path: str, out_dir: str) -> List[str]:
    """
    Extract all embedded images from a PDF and save them to `out_dir`.

    Args:
        pdf_path: Path to the input PDF.
        out_dir: Directory where extracted images will be saved.
                 Created if it doesn't exist.

    Returns:
        List of file paths for the saved images.

    Raises:
        FileNotFoundError: If the PDF does not exist.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    os.makedirs(out_dir, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    saved_paths = []
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        if not image_list:
            continue

        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception as e:
                print(f"[WARN] Failed to extract image xref={xref} "
                      f"on page {page_index + 1}: {e}")
                continue

            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            filename = f"{pdf_name}_page{page_index + 1}_img{img_index}.{image_ext}"
            out_path = os.path.join(out_dir, filename)

            with open(out_path, "wb") as f:
                f.write(image_bytes)

            saved_paths.append(out_path)

    doc.close()
    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Extract images from a PDF.")
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    parser.add_argument("--out-dir", default="extracted_images",
                         help="Directory to save extracted images (default: extracted_images)")
    args = parser.parse_args()

    try:
        paths = extract_images_from_pdf(args.pdf_path, args.out_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not paths:
        print("[INFO] No images found in this PDF.")
        return

    print(f"[INFO] Extracted {len(paths)} image(s):")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
