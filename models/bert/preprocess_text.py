import re
import nltk
from nltk.tokenize import sent_tokenize
from typing import List
from pathlib import Path

def setup_nltk():
    """Setup NLTK resources with proper error handling."""
    try:
        nltk_data_dir = Path.home() / "nltk_data"
        nltk_data_dir.mkdir(exist_ok=True)
        nltk.data.path.append(str(nltk_data_dir))

        resources = [
            "punkt",
            "averaged_perceptron_tagger",
            "tokenizers/punkt",
            "punkt_tab",
        ]

        for resource in resources:
            try:
                nltk.data.find(resource)
            except LookupError:
                print(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, download_dir=str(nltk_data_dir), quiet=False)

        # Verify punkt is properly installed
        from nltk.tokenize import PunktSentenceTokenizer
        tokenizer = PunktSentenceTokenizer()
        test_text = "This is a test. This is another test."
        tokenizer.tokenize(test_text)
        print("NLTK resources successfully installed and verified.")
    except Exception as e:
        print(f"Error setting up NLTK: {str(e)}")
        raise

def preprocess_lecture_text(text: str) -> str:
    """
    Clean and preprocess lecture transcript text.
    """
    if not isinstance(text, str):
        return ""

    # Remove speaker labels and timestamps
    text = re.sub(r"\[Speaker \d+\]|\[\d+:\d+\]", "", text)

    # Remove multiple spaces and newlines
    text = " ".join(text.split())

    # Remove very short sentences (likely noise)
    sentences = [s for s in sent_tokenize(text) if len(s.split()) > 3]

    return " ".join(sentences)