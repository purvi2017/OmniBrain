import os
import uuid


UPLOAD_DIR = "uploads"


def save_file(file_content: bytes) -> tuple[str, str, int]:
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    unique_filename = f"{uuid.uuid4()}.pdf"

    file_path = os.path.join(
        UPLOAD_DIR,
        unique_filename
    )

    with open(file_path, "wb") as buffer:
        buffer.write(file_content)

    file_size = len(file_content)

    return unique_filename, file_path, file_size