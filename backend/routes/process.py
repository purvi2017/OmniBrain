from fastapi import APIRouter, HTTPException

from services.document_service import process_document


router = APIRouter(
    prefix="/process",
    tags=["Document Processing"]
)


@router.post(
    "/{document_id}",
    summary="Process a document",
    description=(
        "Validates the document ID, locates the uploaded PDF, "
        "and prepares the document for future AI processing."
    ),
    responses={
        400: {"description": "Invalid document file"},
        404: {"description": "Document not found"},
        500: {"description": "Document processing failed"}
    }
)
def process_document_endpoint(document_id: str):
    try:
        document = process_document(document_id)

        return {
            "status": "success",
            "message": "Document is ready for processing",
            "document": document
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Document processing failed"
        )