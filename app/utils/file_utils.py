import os
import tempfile
from fastapi import UploadFile

async def save_upload_to_temp(file: UploadFile) -> str:
    """Saves an uploaded file to a temporary location and returns the file path."""
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        return tmp.name
