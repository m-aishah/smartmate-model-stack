import logging
import torch
from transformers import pipeline
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_bert_summarizer(model_name: str = "philschmid/bart-large-cnn-samsum") -> Optional[pipeline]:
    """
    Load the BERT summarization model with proper error handling.
    """
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.warning("Running on CPU - processing may be slower")

        model = pipeline(
            "summarization", model=model_name, device=device, framework="pt"
        )
        logger.info(f"Model loaded successfully on {device}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise