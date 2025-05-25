from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.models.transcription_model import WhisperTranscriber
from app.models.summarization_model import BertSummarizer
from app.services.google_cloud.translate_api import GoogleTranslateAPI
from app.services.assembly_transcriber import AssemblyTranscriber
from starlette.concurrency import run_in_threadpool
from app.utils.file_utils import save_upload_to_temp
import os

router = APIRouter()

@router.post("/audio/file")
async def process_audio_file(
    file: UploadFile = File(...),
    current_language: str = Form("en"),
    target_language: str = Form("tr"),
    model: str = Form("assembly")  # Optional model selection
):
    try:
        # Save uploaded file to temp location
        tmp_path = await save_upload_to_temp(file)

        audio_info = {"file_path": tmp_path}

        # --- Transcription ---
        if model == "whisper":
            transcriber = WhisperTranscriber()
            transcription = transcriber.transcribe(audio_info)
        else:
            transcriber = AssemblyTranscriber()
            transcription = await run_in_threadpool(transcriber.transcribe, audio_info)

        # --- Summarization ---
        summarizer = BertSummarizer()
        summary_result = summarizer.process_lecture(transcription)
        if summary_result["error"]:
            raise HTTPException(status_code=400, detail=summary_result["error"])
        summary = summary_result["detailed_summary"]

        # --- Translation ---
        translator = GoogleTranslateAPI()
        translated_summary = translator.translate_text(summary, target_language)

        return {
            "transcription": transcription,
            "summary": summary,
            "translated_summary": translated_summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
