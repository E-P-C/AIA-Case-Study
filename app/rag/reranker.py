from __future__ import annotations

from collections import Counter

from app.rag.retriever import RetrievedChunk, tokenize


class LexicalReranker:
    """Lightweight lexical reranker (no extra model download).

    Scores candidates by query-term overlap + exact phrase boost.
    Toggle-able via config to support sensitivity analysis.
    """

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        q_tokens = tokenize(query)
        q_counts = Counter(q_tokens)
        q_lower = query.lower().strip()

        rescored: list[RetrievedChunk] = []
        for c in candidates:
            doc_tokens = tokenize(c.text)
            if not doc_tokens:
                lexical = 0.0
            else:
                overlap = sum(min(q_counts[t], doc_tokens.count(t)) for t in q_counts)
                lexical = overlap / (sum(q_counts.values()) + 1e-9)
            phrase_boost = 0.15 if q_lower and q_lower in c.text.lower() else 0.0
            # Blend original hybrid score with lexical signal
            new_score = 0.65 * c.similarity + 0.35 * lexical + phrase_boost
            rescored.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    source=c.source,
                    file_type=c.file_type,
                    page=c.page,
                    similarity=float(new_score),
                )
            )
        rescored.sort(key=lambda x: x.similarity, reverse=True)
        return rescored[:top_k]
