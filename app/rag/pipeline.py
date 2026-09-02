from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, get_settings
from app.observability.logging import JsonlRequestLogger
from app.rag.embeddings import get_embedding_model
from app.rag.generator import LlamaCppGenerator, get_generator
from app.rag.reranker import LexicalReranker
from app.rag.retriever import RetrievedChunk, VectorIndex
from app.rag.session import SessionStore
from app.security.pii import detect_and_redact_pii, should_reject_for_pii
from app.security.prompt_injection import detect_prompt_injection


REFUSAL_LOW_SIM = (
    "There is no relevant information in the current knowledge base "
    "(retrieval similarity below threshold). I can only answer questions "
    "grounded in indexed AKP handbook, compliance, technical, and architecture documents."
)
REFUSAL_EMPTY = (
    "No documents were retrieved for this question. Current capability covers only "
    "the ingested internal knowledge corpus. Please rephrase or ask about indexed topics."
)
REFUSAL_PII = (
    "Your query appears to contain PII (e.g. phone, email, ID). "
    "Per AKP compliance policy, the RAG QA service rejects requests that carry PII."
)
REFUSAL_INJECTION = (
    "Your request was blocked by prompt-injection defense. "
    "Please rephrase without instructions that attempt to override system rules."
)


@dataclass
class QAResult:
    answer: str
    session_id: str
    citations: list[dict]
    refused: bool
    refusal_reason: str | None
    latency_ms: int
    token_in: int
    token_out: int
    retrieved_chunks: list[dict]
    request_id: str


class RAGPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.logger = JsonlRequestLogger(self.settings.log_dir)
        self.sessions = SessionStore(
            idle_timeout_sec=self.settings.session_idle_timeout_sec,
            max_history_turns=self.settings.max_history_turns,
        )
        self.reranker = LexicalReranker()
        self._index: VectorIndex | None = None
        self._generator: LlamaCppGenerator | None = None

    def load_index(self) -> VectorIndex:
        if self._index is None:
            index_dir = Path(self.settings.index_dir)
            if not (index_dir / "chunks.json").exists():
                raise FileNotFoundError(
                    f"Index not found in {index_dir}. Run: python scripts/ingest_docs.py"
                )
            self._index = VectorIndex.load(index_dir)
        return self._index

    def load_generator(self) -> LlamaCppGenerator:
        if self._generator is None:
            self._generator = get_generator(
                str(self.settings.model_path),
                self.settings.n_ctx,
                self.settings.n_threads,
                self.settings.n_gpu_layers,
                self.settings.repeat_penalty,
            )
        return self._generator

    def _format_contexts(self, chunks: list[RetrievedChunk]) -> list[str]:
        blocks = []
        for c in chunks:
            blocks.append(
                f"[source:{c.source},p{c.page}|type:{c.file_type}|score:{c.similarity:.3f}]\n{c.text}"
            )
        return blocks

    def ask(
        self,
        question: str,
        *,
        session_id: str | None = None,
        top_k: int | None = None,
        reranker_enabled: bool | None = None,
        temperature: float | None = None,
    ) -> QAResult:
        t0 = time.perf_counter()
        settings = self.settings
        top_k = settings.top_k if top_k is None else top_k
        reranker_enabled = (
            settings.reranker_enabled if reranker_enabled is None else reranker_enabled
        )
        temperature = settings.temperature if temperature is None else temperature

        session = self.sessions.get_or_create(session_id)
        injection = detect_prompt_injection(question)
        pii_hit = should_reject_for_pii(question)

        def _finalize(
            *,
            answer: str,
            refused: bool,
            reason: str | None,
            chunks: list[RetrievedChunk],
            token_in: int = 0,
            token_out: int = 0,
            status: str = "ok",
        ) -> QAResult:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            citations = [
                {
                    "source": c.source,
                    "page": c.page,
                    "file_type": c.file_type,
                    "similarity": round(c.similarity, 4),
                }
                for c in chunks
            ]
            request_id = self.logger.log_request(
                {
                    "session_id": session.session_id,
                    "user_query": question,
                    "retrieved_chunks": [c.to_dict() for c in chunks],
                    "answer": answer,
                    "token_in": token_in,
                    "token_out": token_out,
                    "latency_ms": latency_ms,
                    "top_k": top_k,
                    "reranker_enabled": reranker_enabled,
                    "temperature": temperature,
                    "prompt_injection_detected": injection.detected,
                    "pii_detected": pii_hit or detect_and_redact_pii(question).detected,
                    "refused": refused,
                    "refusal_reason": reason,
                    "status": status,
                }
            )
            if not refused:
                self.sessions.append_turn(session.session_id, "user", question)
                self.sessions.append_turn(session.session_id, "assistant", answer)
            return QAResult(
                answer=answer,
                session_id=session.session_id,
                citations=citations if not refused else [],
                refused=refused,
                refusal_reason=reason,
                latency_ms=latency_ms,
                token_in=token_in,
                token_out=token_out,
                retrieved_chunks=[c.to_dict() for c in chunks],
                request_id=request_id,
            )

        if injection.detected:
            self.logger.log_security(
                {
                    "event_type": "prompt_injection",
                    "session_id": session.session_id,
                    "detail": f"matched={injection.matched_patterns}; q={question}",
                }
            )
            return _finalize(
                answer=REFUSAL_INJECTION,
                refused=True,
                reason="prompt_injection",
                chunks=[],
                status="blocked",
            )

        if pii_hit:
            self.logger.log_security(
                {
                    "event_type": "pii_detected",
                    "session_id": session.session_id,
                    "detail": f"query rejected for PII: {question}",
                }
            )
            return _finalize(
                answer=REFUSAL_PII,
                refused=True,
                reason="pii_detected",
                chunks=[],
                status="blocked",
            )

        index = self.load_index()
        embedder = get_embedding_model(settings.embedding_model)
        # Retrieve on the current question only. Multi-turn continuity is carried
        # by generation history; rewriting the retrieval query with prior turns
        # biases ranking toward old topics (a new-topic follow-up can answer the
        # previous question instead). Referential follow-ups still contain enough
        # keywords to retrieve, and the model resolves referents via history.
        effective_query = question

        qvec = embedder.embed([effective_query])[0]
        candidates = index.search(effective_query, qvec, top_k=top_k)
        if reranker_enabled:
            chunks = self.reranker.rerank(effective_query, candidates, top_k=top_k)
        else:
            chunks = candidates[:top_k]

        # Context pruning: drop low-confidence chunks relative to the best hit so
        # topically-adjacent noise does not dilute the generation context.
        if chunks:
            best_sim = chunks[0].similarity
            chunks = [c for c in chunks if c.similarity >= best_sim * settings.context_gate_ratio]

        if not chunks:
            return _finalize(
                answer=REFUSAL_EMPTY,
                refused=True,
                reason="no_retrieval",
                chunks=[],
            )

        best = chunks[0].similarity
        if best < settings.similarity_threshold:
            return _finalize(
                answer=REFUSAL_LOW_SIM,
                refused=True,
                reason="low_similarity",
                chunks=chunks,
            )

        generator = self.load_generator()
        gen = generator.generate(
            question=question,
            contexts=self._format_contexts(chunks),
            history=session.history,
            temperature=temperature,
            max_tokens=settings.max_new_tokens,
        )
        return _finalize(
            answer=gen.answer,
            refused=False,
            reason=None,
            chunks=chunks,
            token_in=gen.token_in,
            token_out=gen.token_out,
        )


_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
