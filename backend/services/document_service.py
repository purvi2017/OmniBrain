import os
from fastapi import HTTPException

UPLOAD_DIR = "uploads"


def process_document(document_id: str):
    filename = f"{document_id}.pdf"
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=400,
            detail="Invalid document file"
        )

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF documents can be processed"
        )

    try:
        file_size = os.path.getsize(file_path)

        return {
            "document_id": document_id,
            "filename": filename,
            "path": file_path,
            "size": file_size,
            "status": "ready"
        }

    except OSError:
        raise HTTPException(
            status_code=500,
            detail="Failed to access document"
        )