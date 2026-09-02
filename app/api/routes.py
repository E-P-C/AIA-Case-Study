from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from app.api.schemas import AskRequest, AskResponse, Citation, HealthResponse
from app.config import Settings, get_settings
from app.rag.pipeline import RAGPipeline, get_pipeline
from app.rag.retriever import VectorIndex

router = APIRouter()


# Security scheme so /docs shows an Authorize button. auto_error=False keeps the
# existing explicit 401 (we own the error message) instead of FastAPI's auto-401.
api_key_header = APIKeyHeader(name="X-API-Token", auto_error=False)


def require_token(
    x_api_token: str | None = Security(api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    if x_api_token != settings.api_token:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Token")


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    index_path = Path(settings.index_dir) / "chunks.json"
    model_present = Path(settings.model_path).exists()
    chunk_count = 0
    index_loaded = False
    if index_path.exists():
        try:
            idx = VectorIndex.load(settings.index_dir)
            chunk_count = len(idx.chunks)
            index_loaded = True
        except Exception:
            index_loaded = False
    return HealthResponse(
        status="ok",
        index_loaded=index_loaded,
        model_present=model_present,
        chunk_count=chunk_count,
    )


@router.post("/ask", response_model=AskResponse, dependencies=[Depends(require_token)])
def ask(
    body: AskRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> AskResponse:
    try:
        result = pipeline.ask(
            body.question,
            session_id=body.session_id,
            top_k=body.top_k,
            reranker_enabled=body.reranker_enabled,
            temperature=body.temperature,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"internal error: {exc}") from exc

    return AskResponse(
        answer=result.answer,
        session_id=result.session_id,
        citations=[Citation(**c) for c in result.citations],
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        latency_ms=result.latency_ms,
        token_in=result.token_in,
        token_out=result.token_out,
        retrieved_chunks=result.retrieved_chunks,
        request_id=result.request_id,
    )
