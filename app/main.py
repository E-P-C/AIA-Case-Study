from __future__ import annotations

import logging

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app import __version__
from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(
    title="AKP Internal RAG QA Service",
    version=__version__,
    description=(
        "Retrieval-augmented QA over AKP bilingual knowledge corpus. "
        "Uses embedded llama-cpp-python (in-process GGUF) — no external inference server."
    ),
)
app.include_router(router, prefix="/v1")


STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/chat", include_in_schema=False)
def chat_page() -> FileResponse:
    """Lightweight browser chat UI backed by POST /v1/ask."""
    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/")
def root() -> dict:
    return {
        "service": "akp-rag-qa",
        "version": __version__,
        "docs": "/docs",
        "chat": "/chat",
        "health": "/v1/health",
        "ask": "POST /v1/ask",
    }
