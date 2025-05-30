import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.models.summarization_model import BertSummarizer
from app.utils.file_utils import save_upload_to_temp
from starlette.concurrency import run_in_threadpool
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/text/")
async def summarize_text(text: str):
    try:
        summarizer = BertSummarizer()
        result = summarizer.process_lecture(text)
        if result["error"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/audio/file")
async def summarize_audio_file(
    file: UploadFile = File(...),
    model: str = Form("assembly")
):
    try:
        # Save the uploaded file temporarily
        tmp_path = await save_upload_to_temp(file)

        # Prepare audio_info
        audio_info = {"file_path": tmp_path}

        # Choose transcriber based on model
        if model == "whisper":
            from app.models.transcription_model import WhisperTranscriber
            transcriber = WhisperTranscriber()
            transcription = transcriber.transcribe(audio_info)
        else:
            from app.services.assembly_transcriber import AssemblyTranscriber
            transcriber = AssemblyTranscriber()
            transcription = await run_in_threadpool(transcriber.transcribe, audio_info)

        logger.info(f"Transcription completed for {file.filename}")

        # Summarize the transcription
        summarizer = BertSummarizer()
        summary_result = summarizer.process_lecture(transcription)
        if summary_result["error"]:
            raise HTTPException(status_code=400, detail=summary_result["error"])

        return {
            "transcription": transcription,
            "summary": summary_result["detailed_summary"],
            "brief_summary": summary_result["brief_summary"],
            "key_points": summary_result["key_points"],
        }

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
