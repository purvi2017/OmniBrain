from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, HTTPException

from utils.validators import validate_pdf
from services.file_service import save_file
from schemas.upload import UploadResponse


router = APIRouter()


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a PDF document",
    description=(
        "Uploads a PDF document, validates its file type and size, "
        "generates a unique document ID, and stores the document."
    ),
    responses={
        400: {"description": "No file selected, invalid file type, or empty file"},
        413: {"description": "File exceeds the maximum allowed size"},
        500: {"description": "File upload failed due to a server error"}
    }
)
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_content = await file.read()

        validate_pdf(file, len(file_content))

        (
            document_id,
            filename,
            file_path,
            file_size
        ) = save_file(file_content)

        upload_time = datetime.now(timezone.utc).isoformat()

        return {
            "status": "success",
            "document_id": document_id,
            "filename": filename,
            "size": file_size,
            "upload_time": upload_time
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="File upload failed"
        ) from exc