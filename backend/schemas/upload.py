from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    status: str = Field(
        ...,
        description="Upload processing status."
    )
    document_id: str = Field(
        ...,
        description="Unique ID generated for the uploaded document."
    )
    filename: str = Field(
        ...,
        description="Stored filename of the uploaded document."
    )
    size: int = Field(
        ...,
        description="Size of the uploaded file in bytes."
    )
    upload_time: str = Field(
        ...,
        description="UTC timestamp of when the file was uploaded."
    )