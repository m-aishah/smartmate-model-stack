from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List, Dict

def create_smart_chunks(text: str, chunk_size: int = 800, overlap_size: int = 100) -> List[Dict[str, str]]:
    """
    Create overlapping chunks with context preservation.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for i, sentence in enumerate(sentences):
        words = word_tokenize(sentence)

        if current_length + len(words) > chunk_size:
            chunk_text = " ".join([s for s in current_chunk])

            if i < len(sentences) - 1:
                next_context = " ".join(sentences[i : i + 3])
                chunks.append({"text": chunk_text, "next_context": next_context})
            else:
                chunks.append({"text": chunk_text, "next_context": ""})

            overlap_sentences = current_chunk[-3:]
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(word_tokenize(s)) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += len(words)

    if current_chunk:
        chunks.append({"text": " ".join(current_chunk), "next_context": ""})

    return chunks