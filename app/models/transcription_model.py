import os
import logging
from typing import Optional
from models.whisper_pretrained.load_whisper import (
    load_whisper_model,
    transcribe_audio_to_text,
)

logger = logging.getLogger(__name__)

def assert_audio_duration(audio_file_path: str) -> float:
    try:
        import soundfile as sf

        audio_info = sf.info(audio_file_path)
        audio_duration = audio_info.duration
        logger.debug(f"Audio duration for {audio_file_path}: {audio_duration:.2f} seconds")
        return audio_duration
    except Exception as e:
        logger.exception(f"Failed to get audio duration for {audio_file_path}")
        raise


class WhisperTranscriber:
    def __init__(self, model_name: str = "openai/whisper-base"):
        logger.info(f"Loading Whisper model: {model_name}")
        self.whisper_model = load_whisper_model(model_name)
        logger.info("Whisper model loaded successfully")

    def transcribe(self, audio_info: dict):
        if not isinstance(audio_info, dict):
            logger.error("audio_info is not a dictionary")
            raise TypeError("audio_info must be a dictionary")
        if "file_path" not in audio_info:
            logger.error("Missing 'file_path' in audio_info")
            raise ValueError("audio_info must contain 'file_path'")

        audio_file_path = audio_info["file_path"]

        if not os.path.isfile(audio_file_path):
            logger.error(f"File not found: {audio_file_path}")
            raise FileNotFoundError(f"Audio file '{audio_file_path}' not found")

        if "duration" not in audio_info:
            logger.debug("Duration not provided, calculating...")
            audio_info["duration"] = assert_audio_duration(audio_file_path)

        try:
            logger.info(f"Transcribing audio: {audio_file_path}")
            transcription = transcribe_audio_to_text(self.whisper_model, audio_info)
            logger.debug(f"Transcription result: {transcription}")
            return transcription
        except Exception as e:
            logger.exception(f"Error during transcription for file {audio_file_path}")
            raise

    def test_transcription(self, audio_file_path):
        logger.info(f"Testing transcription for: {audio_file_path}")
        audio_info = {"file_path": audio_file_path}
        transcription = self.transcribe(audio_info)
        logger.info(f"Test transcription output: {transcription}")
        print(transcription)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)  # For direct script testing
    audio_info = {
        "file_path": "/home/maishah/Graduation Project/models_stack/data/raw-audio/sample_short.wav",
        "duration": 60,
    }

    transcriber = WhisperTranscriber()
    transcriber.test_transcription(audio_info["file_path"])
