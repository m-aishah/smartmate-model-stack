from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.models.transcription_model import WhisperTranscriber
from app.services.assembly_transcriber import AssemblyTranscriber
from app.utils.file_utils import save_upload_to_temp
from starlette.concurrency import run_in_threadpool
import tempfile
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/audio/file")
async def transcribe_audio_file(
    file: UploadFile = File(...),
    model: str = Form("assembly")
):
    try:
        logger.info(f"Received file for transcription: {file.filename}")
        logger.info(f"Transcription model requested: {model}")

        # Save uploaded audio to a temp file
        tmp_path = await save_upload_to_temp(file)
        logger.debug(f"Temporary file created at: {tmp_path}")

        # Prepare audio_info dict
        audio_info = {"file_path": tmp_path}

        # Select transcriber dynamically
        if model == "whisper":
            transcriber = WhisperTranscriber()
            transcription = transcriber.transcribe(audio_info)  # Whisper is sync
        else:
            transcriber = AssemblyTranscriber()
            transcription = await run_in_threadpool(transcriber.transcribe, audio_info)

        logger.info("Transcription completed successfully")
        logger.info(f"Transcription result: {transcription}")

        return {"transcription": transcription}

    except Exception as e:
        logger.exception("Transcription failed due to an unexpected error")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.debug(f"Temporary file deleted: {tmp_path}")
