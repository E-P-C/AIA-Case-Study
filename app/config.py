from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_path: Path = ROOT / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    index_dir: Path = ROOT / "data" / "index"
    raw_dir: Path = ROOT / "data" / "raw"
    log_dir: Path = ROOT / "logs"

    top_k: int = 4
    similarity_threshold: float = 0.45
    # Context pruning: keep chunks scoring >= (best_similarity * context_gate_ratio).
    # Drops topically-adjacent noise (e.g. overlapping benefits PDFs) from the
    # generation context so answers stay grounded in the most relevant source.
    context_gate_ratio: float = 0.80
    reranker_enabled: bool = True
    temperature: float = 0.2
    # Suppresses repetition loops (a 1.5B model can echo a repetitive
    # context block until max_new_tokens). Raises latency and hurts quality.
    repeat_penalty: float = 1.3
    max_new_tokens: int = 512
    n_ctx: int = 4096
    n_gpu_layers: int = 0
    n_threads: int = 4

    session_idle_timeout_sec: int = 1800
    api_token: str = "akp-demo-token"

    cost_per_1m_input_tokens: float = 0.10
    cost_per_1m_output_tokens: float = 0.30

    chunk_size: int = 500
    chunk_overlap: int = 80
    max_history_turns: int = 6


@lru_cache
def get_settings() -> Settings:
    return Settings()
