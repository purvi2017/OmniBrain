from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import uuid

app = FastAPI(
    title="OmniBrain API",
    version="1.0.0"
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "🚀 OmniBrain API Running Successfully!"
    }


@app.get("/health")
def health():
    return {
        "status": "Healthy"
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # 1. Check if a file was selected
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file selected"
            )

        # 2. PDF file validation
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )

        # 3. Generate a unique filename using UUID
        unique_filename = f"{uuid.uuid4()}.pdf"

        # 4. Create the upload path
        file_path = os.path.join(
            UPLOAD_DIR,
            unique_filename
        )

        # 5. Read and save the uploaded file
        file_content = await file.read()

        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        # 6. Return the required response
        return {
            "status": "success",
            "filename": unique_filename,
            "path": file_path
        }

    except HTTPException:
        # Re-raise our intentional HTTP errors
        raise

    except Exception:
        # Handle unexpected upload errors
        raise HTTPException(
            status_code=500,
            detail="File upload failed"
        )