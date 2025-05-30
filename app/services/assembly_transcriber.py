import os
import logging
import assemblyai as aai

logger = logging.getLogger(__name__)

class AssemblyTranscriber:
    def __init__(self, api_key: str = None):
        if not api_key:
            api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            raise ValueError("AssemblyAI API key must be provided")
        
        # logging.info(api_key)
        aai.settings.api_key = api_key
        self.transcriber = aai.Transcriber(config=aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best
        ))
        logger.info("AssemblyAI Transcriber initialized")

    def transcribe(self, audio_info: dict):
        if not isinstance(audio_info, dict):
            raise TypeError("audio_info must be a dictionary")

        if "file_path" not in audio_info:
            raise ValueError("audio_info must contain 'file_path'")

        audio_file_path = audio_info["file_path"]

        if not os.path.isfile(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        try:
            logger.info(f"Transcribing local file via AssemblyAI: {audio_file_path}")
            transcript = self.transcriber.transcribe(audio_file_path)
            logger.debug(f"Transcription result: {transcript}")
            if transcript.status == "error":
                raise RuntimeError(f"Transcription failed: {transcript.error}")
            return transcript.text
            # return "This is a test transcription from AssemblyAI."
        except Exception as e:
            logger.exception(f"Error during transcription for file {audio_file_path}")
            raise

    def test_transcription(self, audio_file_path):
        audio_info = {"file_path": audio_file_path}
        transcription = self.transcribe(audio_info)
        print(transcription)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    transcriber = AssemblyTranscriber(api_key=os.getenv("ASSEMBLYAI_API_KEY"))
    transcriber.test_transcription("/tests/audios/test-audio-16min.wav")
