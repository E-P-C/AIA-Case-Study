from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.security.pii import detect_and_redact_pii


class JsonlRequestLogger:
    """Structured request logger for latency + generation diagnosis."""

    def __init__(self, log_dir: Path | None = None) -> None:
        settings = get_settings()
        self.log_dir = Path(log_dir or settings.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / "requests.jsonl"
        self.security_path = self.log_dir / "security_audit.jsonl"
        self._logger = logging.getLogger("rag.request")

    def _write(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_request(self, event: dict[str, Any]) -> str:
        request_id = event.get("request_id") or str(uuid.uuid4())
        raw_query = str(event.get("user_query", ""))
        redacted_query = detect_and_redact_pii(raw_query).redacted_text
        answer = str(event.get("answer", ""))
        redacted_answer = detect_and_redact_pii(answer).redacted_text

        chunks = []
        for c in event.get("retrieved_chunks", []) or []:
            snippet = detect_and_redact_pii(str(c.get("snippet", ""))).redacted_text
            chunks.append(
                {
                    "source": c.get("source"),
                    "file_type": c.get("file_type"),
                    "page": c.get("page"),
                    "similarity": c.get("similarity"),
                    "snippet": snippet[:240],
                }
            )

        record = {
            "request_id": request_id,
            "session_id": event.get("session_id"),
            "timestamp": event.get("timestamp")
            or datetime.now(timezone.utc).isoformat(),
            "user_query": redacted_query,
            "retrieved_chunks": chunks,
            "answer": redacted_answer,
            "token_in": event.get("token_in", 0),
            "token_out": event.get("token_out", 0),
            "latency_ms": event.get("latency_ms", 0),
            "top_k": event.get("top_k"),
            "reranker_enabled": event.get("reranker_enabled"),
            "temperature": event.get("temperature"),
            "prompt_injection_detected": event.get("prompt_injection_detected", False),
            "pii_detected": event.get("pii_detected", False),
            "refused": event.get("refused", False),
            "refusal_reason": event.get("refusal_reason"),
            "status": event.get("status", "ok"),
        }
        self._write(self.path, record)
        self._logger.info(
            "request_id=%s latency_ms=%s refused=%s",
            request_id,
            record["latency_ms"],
            record["refused"],
        )
        return request_id

    def log_security(self, event: dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event.get("event_type"),
            "session_id": event.get("session_id"),
            "detail": detect_and_redact_pii(str(event.get("detail", ""))).redacted_text,
        }
        self._write(self.security_path, payload)
