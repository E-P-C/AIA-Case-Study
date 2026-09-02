from __future__ import annotations

import re
from dataclasses import dataclass


INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"forget\s+(your|all)\s+(rules|instructions|prompt)",
        r"rewrite\s+(your\s+)?system\s+prompt",
        r"you\s+are\s+now\s+(dan|unrestricted|jailbroken)",
        r"disclose\s+(your\s+)?system\s+prompt",
        r"忽略(之前|以上|先前)?(的)?(所有)?(指令|规则|提示)",
        r"忘记(你的|所有)?(规则|指令|提示词)",
        r"改写(你的)?系统提示词",
        r"输出(你的)?系统提示",
        r"jailbreak",
    ]
]


@dataclass
class InjectionResult:
    detected: bool
    matched_patterns: list[str]
    sanitized_query: str


def detect_prompt_injection(text: str) -> InjectionResult:
    matched: list[str] = []
    for pat in INJECTION_PATTERNS:
        if pat.search(text):
            matched.append(pat.pattern)
    # Soft sanitize: strip matched spans for downstream logging only.
    sanitized = text
    for pat in INJECTION_PATTERNS:
        sanitized = pat.sub("[BLOCKED_INSTRUCTION]", sanitized)
    return InjectionResult(
        detected=bool(matched),
        matched_patterns=matched,
        sanitized_query=sanitized.strip(),
    )
