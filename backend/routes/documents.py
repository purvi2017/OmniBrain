from fastapi import APIRouter, HTTPException
import os
import logging

router = APIRouter()

UPLOAD_DIR = "uploads"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.get("/documents")
def get_documents():
    documents = []

    if os.path.exists(UPLOAD_DIR):
        for filename in os.listdir(UPLOAD_DIR):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(UPLOAD_DIR, filename)

                documents.append({
                    "filename": filename,
                    "path": file_path,
                    "size": os.path.getsize(file_path)
                })

    return {
        "status": "success",
        "documents": documents
    }


@router.get("/document/{id}")
def get_document(id: str):
    file_path = os.path.join(UPLOAD_DIR, id)

    if not os.path.exists(file_path):
        logger.warning(f"Document not found: {id}")
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    logger.info(f"Document details requested: {id}")

    return {
        "status": "success",
        "document": {
            "document_id": id,
            "filename": id,
            "path": file_path,
            "size": os.path.getsize(file_path)
        }
    }


@router.delete("/document/{id}")
def delete_document(id: str):
    file_path = os.path.join(UPLOAD_DIR, id)

    if not os.path.exists(file_path):
        logger.warning(f"Document not found for deletion: {id}")
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    try:
        os.remove(file_path)

        logger.info(f"Document deleted successfully: {id}")

        return {
            "status": "success",
            "message": "Document deleted successfully",
            "document_id": id
        }

    except Exception as e:
        logger.error(f"Error deleting document {id}: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Failed to delete document"
        )