from fastapi import FastAPI
from app.routes import register_routes
from fastapi.middleware.cors import CORSMiddleware
from app.logging_config import setup_logging
from dotenv import load_dotenv
import os

load_dotenv()
print("AssemblyAI API Key:", os.getenv("ASSEMBLYAI_API_KEY"))
setup_logging()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # TODO: Restrict to bckend api after deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
register_routes(app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
