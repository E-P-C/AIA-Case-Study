from __future__ import annotations

import logging

from fastapi import FastAPI

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


@app.get("/")
def root() -> dict:
    return {
        "service": "akp-rag-qa",
        "version": __version__,
        "docs": "/docs",
        "health": "/v1/health",
        "ask": "POST /v1/ask",
    }
