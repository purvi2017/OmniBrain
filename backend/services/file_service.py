import os
import uuid

UPLOAD_DIR = "uploads"


def save_file(file_content: bytes) -> tuple[str, str, str, int]:
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    document_id = str(uuid.uuid4())
    unique_filename = f"{document_id}.pdf"

    file_path = os.path.join(
        UPLOAD_DIR,
        unique_filename
    )

    with open(file_path, "wb") as buffer:
        buffer.write(file_content)

    file_size = len(file_content)

    return document_id, unique_filename, file_path, file_size