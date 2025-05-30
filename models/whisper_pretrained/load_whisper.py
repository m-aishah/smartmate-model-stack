import torch
import logging
from transformers import pipeline
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def load_whisper_model(
    model_name: str = "openai/whisper-base",
) -> Optional[pipeline]:
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper model '{model_name}' on {device}")
        whisper_model = pipeline(
            "automatic-speech-recognition", model=model_name, device=device
        )
        logger.info("Whisper model loaded successfully")
        return whisper_model
    except Exception as e:
        logger.exception(f"Failed to load Whisper model '{model_name}'")
        raise


def transcribe_short_audio_to_text(
    speech_recognition_model, audio_file_path: str
) -> str:
    try:
        logger.debug(f"Transcribing short audio file: {audio_file_path}")
        transcription = speech_recognition_model(audio_file_path)["text"]
        logger.debug("Short audio transcription complete")
        return transcription.strip()
    except Exception as e:
        logger.exception(f"Error transcribing short audio file: {audio_file_path}")
        raise

def transcribe_long_audio_to_text(
    speech_recognition_model, audio_file_path: str
) -> str:
    try:
        logger.debug(f"Transcribing long audio file: {audio_file_path}")
        transcription = speech_recognition_model(
            audio_file_path, max_new_tokens=256, chunk_length_s=30, batch_size=8
        )["text"]
        logger.debug("Long audio transcription complete")
        return transcription.strip()
    except Exception as e:
        logger.exception(f"Error transcribing long audio file: {audio_file_path}")
        raise

def transcribe_audio_to_text(speech_recognition_model, audio_info: Dict) -> str:
    try:
        audio_duration = audio_info["duration"]
        audio_file_path = audio_info["file_path"]

        logger.info(f"Starting transcription for file: {audio_file_path} (duration: {audio_duration}s)")
        if audio_duration <= 30:
            return transcribe_short_audio_to_text(speech_recognition_model, audio_file_path)
        else:
            return transcribe_long_audio_to_text(speech_recognition_model, audio_file_path)
    except Exception as e:
        logger.exception(f"Error during transcription for: {audio_info.get('file_path')}")
        raise

def test_transcribe_audio(audio_info):
    logging.basicConfig(level=logging.DEBUG)  # For standalone testing
    whisper_model = load_whisper_model()
    transcription = transcribe_audio_to_text(whisper_model, audio_info)
    logger.info(f"Test transcription: {transcription}")
    print(transcription)

if __name__ == "__main__":
    audio_info = {
        "file_path": "/home/maishah/Graduation Project/models_stack/data/raw-audio/sample_short.wav",
        "duration": 50,
    }
    test_transcribe_audio(audio_info)
