from fastapi import FastAPI
from app.controllers import summarization, transcription, translation, process_audio

def register_routes(app: FastAPI):
    app.include_router(transcription.router, prefix="/transcribe")
    app.include_router(summarization.router, prefix="/summarize")
    app.include_router(translation.router, prefix="/translate")
    app.include_router(process_audio.router, prefix="/process")