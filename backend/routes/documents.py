from fastapi import APIRouter, HTTPException
import os
import logging


router = APIRouter()

UPLOAD_DIR = "uploads"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.get(
    "/documents",
    summary="List all uploaded documents",
    description="Returns details of all PDF documents stored in the uploads directory."
)
def get_documents():
    documents = []

    if not os.path.exists(UPLOAD_DIR):
        return {
            "status": "success",
            "documents": []
        }

    for filename in os.listdir(UPLOAD_DIR):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(UPLOAD_DIR, filename)

            document_id = os.path.splitext(filename)[0]

            documents.append({
                "document_id": document_id,
                "filename": filename,
                "path": file_path,
                "size": os.path.getsize(file_path)
            })

    return {
        "status": "success",
        "documents": documents
    }


@router.get(
    "/document/{document_id}",
    summary="Get document details",
    description="Returns details of a specific document using its unique document ID."
)
def get_document(document_id: str):
    filename = f"{document_id}.pdf"
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        logger.warning(f"Document not found: {document_id}")

        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    logger.info(f"Document details requested: {document_id}")

    return {
        "status": "success",
        "document": {
            "document_id": document_id,
            "filename": filename,
            "path": file_path,
            "size": os.path.getsize(file_path)
        }
    }


@router.delete(
    "/document/{document_id}",
    summary="Delete a document",
    description="Deletes a specific PDF document using its unique document ID."
)
def delete_document(document_id: str):
    filename = f"{document_id}.pdf"
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        logger.warning(
            f"Document not found for deletion: {document_id}"
        )

        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    try:
        os.remove(file_path)

        logger.info(
            f"Document deleted successfully: {document_id}"
        )

        return {
            "status": "success",
            "message": "Document deleted successfully",
            "document_id": document_id
        }

    except Exception as e:
        logger.error(
            f"Error deleting document {document_id}: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to delete document"
        )