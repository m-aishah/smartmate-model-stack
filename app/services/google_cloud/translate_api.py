from google.cloud import translate_v2 as translate
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleTranslateAPI:
    """
    A service for translating text using Google Cloud Translation API.
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Google Translate API client.
        If no API key is provided, it will use the default credentials.
        """
        if api_key:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = api_key
        self.client = translate.Client()

    def translate_text(self, text: str, target_language: str = "en") -> str:
        """
        Translate the given text to the target language.
        """
        try:
            result = self.client.translate(text, target_language=target_language)
            return result['translatedText']
        except Exception as e:
            logger.error(f"Error translating text: {e}")
            raise