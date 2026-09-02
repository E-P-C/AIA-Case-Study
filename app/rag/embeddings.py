from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np


def _huggingface_cache_dir() -> Path:
    """Resolve the Hugging Face hub cache directory (respects HF_* overrides)."""
    env = os.environ.get("HF_HUB_CACHE")
    if env:
        return Path(env)
    env = os.environ.get("HF_HOME")
    if env:
        return Path(env) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _model_snapshot_available(model_name: str) -> bool:
    """True when a complete HF snapshot already exists for this model.

    Used to auto-enable offline mode: in network-restricted environments (e.g.
    behind a corporate firewall) the SDK otherwise hangs on an update check
    even though the weights are fully cached.
    """
    org, _, name = model_name.partition("/")
    model_dir = _huggingface_cache_dir() / f"models--{org}--{name}"
    if not model_dir.is_dir():
        return False
    snapshots = model_dir / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


class EmbeddingModel:
    """Thin wrapper so the embedding backend stays swappable."""

    def __init__(self, model_name: str) -> None:
        # Auto-offline fallback when the model is already cached: avoids the
        # Hugging Face update-check hang on machines without HF access.
        if "HF_HUB_OFFLINE" not in os.environ and _model_snapshot_available(model_name):
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


@lru_cache
def get_embedding_model(model_name: str) -> EmbeddingModel:
    return EmbeddingModel(model_name)
