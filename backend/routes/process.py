from fastapi import APIRouter, HTTPException

from services.document_service import process_document


router = APIRouter(
    prefix="/process",
    tags=["Document Processing"]
)


@router.post(
    "/{document_id}",
    summary="Process an uploaded document",
    description=(
        "Processes an uploaded PDF document using its document ID. "
        "The API validates the document ID, checks whether the PDF exists, "
        "extracts text from the PDF, and returns the processing status "
        "and processed document details."
    ),
    responses={
        200: {
            "description": "Document processed successfully"
        },
        400: {
            "description": (
                "Invalid or missing document ID, or invalid document file"
            )
        },
        404: {
            "description": "Document not found"
        },
        500: {
            "description": "Document processing or PDF extraction failed"
        }
    }
)
def process_document_endpoint(document_id: str):
    """
    Process an uploaded PDF document.

    Returns the document details, extracted text,
    and processing status.
    """
    try:
        document = process_document(document_id)

        return {
            "status": "success",
            "processing_status": document["status"],
            "message": "Document processed successfully and is ready for RAG.",
            "document": document
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "message": "Document processing failed"
            }
        )