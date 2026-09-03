import os
import fitz
from fastapi import HTTPException

UPLOAD_DIR = "uploads"


def process_document(document_id: str):
    # Validate document ID
    if not document_id or not document_id.strip():
        raise HTTPException(
            status_code=400,
            detail="document_id is required"
        )

    document_id = document_id.strip()

    # Prevent invalid path/file names
    if "/" in document_id or "\\" in document_id or ".." in document_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid document_id"
        )

    filename = f"{document_id}.pdf"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Check whether the document exists
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {document_id}"
        )

    # Check that it is a valid file
    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=400,
            detail="Invalid document file"
        )

    # Check PDF extension
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF documents can be processed"
        )

    # Processing starts
    processing_status = "processing"

    try:
        file_size = os.path.getsize(file_path)

        # Open PDF and extract text
        pdf_document = fitz.open(file_path)

        text_parts = []

        for page in pdf_document:
            page_text = page.get_text()

            if page_text:
                text_parts.append(page_text)

        pdf_document.close()

        extracted_text = "\n".join(text_parts).strip()

        # Processing completed successfully
        processing_status = "completed"

        return {
            "document_id": document_id,
            "filename": filename,
            "path": file_path,
            "size": file_size,
            "text_length": len(extracted_text),
            "extracted_text": extracted_text,
            "status": processing_status
        }

    except Exception as e:
        # Processing failed
        processing_status = "failed"

        raise HTTPException(
            status_code=500,
            detail={
                "status": processing_status,
                "message": "PDF text extraction failed",
                "error": str(e)
            }
        )