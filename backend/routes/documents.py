from fastapi import APIRouter
import os

router = APIRouter()

UPLOAD_DIR = "uploads"


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