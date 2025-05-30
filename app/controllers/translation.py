from fastapi import APIRouter, HTTPException
from app.services.google_cloud.translate_api import GoogleTranslateAPI
from typing import Optional

router = APIRouter()


@router.post("/text/")
async def translate_text(text: str, target_language: Optional[str] = "en"):
    """
    Translate the given text to the target language.
    """
    try:
        translator = GoogleTranslateAPI()
        # translated_text = translator.translate_text(text, target_language) TODO: Uncomment this line
        translated_text = text # TODO: Remove this line
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))