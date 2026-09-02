from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class SessionState:
    session_id: str
    history: list[dict[str, str]] = field(default_factory=list)
    last_access: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_access = time.time()


class SessionStore:
    def __init__(self, idle_timeout_sec: int = 1800, max_history_turns: int = 6) -> None:
        self.idle_timeout_sec = idle_timeout_sec
        self.max_history_turns = max_history_turns
        self._sessions: dict[str, SessionState] = {}
        self._lock = Lock()

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s.last_access > self.idle_timeout_sec
        ]
        for sid in expired:
            del self._sessions[sid]

    def get_or_create(self, session_id: str | None) -> SessionState:
        with self._lock:
            self._purge_expired()
            if session_id and session_id in self._sessions:
                state = self._sessions[session_id]
                state.touch()
                return state
            sid = session_id or str(uuid.uuid4())
            state = SessionState(session_id=sid)
            self._sessions[sid] = state
            return state

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            state = self._sessions.get(session_id)
            if not state:
                return
            state.history.append({"role": role, "content": content})
            # Keep last N user+assistant pairs
            max_messages = self.max_history_turns * 2
            if len(state.history) > max_messages:
                state.history = state.history[-max_messages:]
            state.touch()
