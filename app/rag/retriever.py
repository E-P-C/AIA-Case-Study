from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.rag.embeddings import EmbeddingModel


TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    file_type: str
    page: int
    similarity: float

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "snippet": self.text,
            "source": self.source,
            "file_type": self.file_type,
            "page": self.page,
            "similarity": round(float(self.similarity), 4),
        }


class VectorIndex:
    def __init__(
        self,
        chunks: list[dict],
        embeddings: np.ndarray,
        embedding_model_name: str,
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)
        self.embedding_model_name = embedding_model_name
        self._bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    @classmethod
    def build(cls, chunks: list[dict], embedder: EmbeddingModel) -> "VectorIndex":
        texts = [c["text"] for c in chunks]
        vectors = embedder.embed(texts)
        return cls(chunks, vectors, embedder.model_name)

    def save(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self.embeddings)
        meta = {
            "embedding_model": self.embedding_model_name,
            "chunks": self.chunks,
        }
        (index_dir / "chunks.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: Path) -> "VectorIndex":
        index_dir = Path(index_dir)
        embeddings = np.load(index_dir / "embeddings.npy")
        meta = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        return cls(meta["chunks"], embeddings, meta["embedding_model"])

    def search(
        self,
        query: str,
        query_vec: np.ndarray,
        *,
        top_k: int = 4,
        dense_weight: float = 0.7,
    ) -> list[RetrievedChunk]:
        if len(self.chunks) == 0:
            return []

        q = query_vec.astype(np.float32).reshape(-1)
        dense_scores = self.embeddings @ q

        bm25_scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)
        if bm25_scores.max() > 0:
            bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)
        else:
            bm25_norm = bm25_scores

        # Map dense cosine ([-1,1]) roughly into [0,1]
        dense_norm = (dense_scores + 1.0) / 2.0
        hybrid = dense_weight * dense_norm + (1.0 - dense_weight) * bm25_norm

        k = min(top_k, len(self.chunks))
        # Over-fetch for optional reranker
        fetch_n = min(len(self.chunks), max(k * 3, k))
        idxs = np.argpartition(-hybrid, fetch_n - 1)[:fetch_n]
        idxs = idxs[np.argsort(-hybrid[idxs])]

        results: list[RetrievedChunk] = []
        for i in idxs:
            c = self.chunks[int(i)]
            results.append(
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    source=c["source"],
                    file_type=c["file_type"],
                    page=int(c["page"]),
                    similarity=float(hybrid[int(i)]),
                )
            )
        return results
