from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, HTTPException

from utils.validators import validate_pdf
from services.file_service import save_file


router = APIRouter()


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_content = await file.read()

        validate_pdf(file, len(file_content))

        filename, file_path, file_size = save_file(file_content)

        upload_time = datetime.now(timezone.utc).isoformat()

        return {
            "status": "success",
            "filename": filename,
            "size": file_size,
            "upload_time": upload_time
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="File upload failed"
        )