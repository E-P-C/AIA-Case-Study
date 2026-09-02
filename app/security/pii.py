from __future__ import annotations

import re
from dataclasses import dataclass


# Prefer explicit phone shapes to avoid false positives on policy numbers (e.g. "2 days").
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s])\d{3,4}[-.\s]\d{4}(?!\d)"
    r"|(?<!\d)1[3-9]\d{9}(?!\d)"
)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
ID_RE = re.compile(r"(?<!\d)(\d{17}[\dXx]|\d{15})(?!\d)")
ADDRESS_HINT_RE = re.compile(
    r"(?:住址|地址|Address)\s*[:：]\s*[^\n,，]{6,80}",
    re.IGNORECASE,
)


@dataclass
class PIIResult:
    detected: bool
    redacted_text: str
    matches: list[str]


def detect_and_redact_pii(text: str) -> PIIResult:
    matches: list[str] = []
    redacted = text

    for pattern, label in (
        (EMAIL_RE, "[REDACTED_EMAIL]"),
        (ID_RE, "[REDACTED_ID]"),
        (PHONE_RE, "[REDACTED_PHONE]"),
        (ADDRESS_HINT_RE, "[REDACTED_ADDRESS]"),
    ):
        found = pattern.findall(redacted)
        if found:
            matches.extend([f if isinstance(f, str) else "".join(f) for f in found])
            redacted = pattern.sub(label, redacted)

    # Heuristic: payment / customer record keywords with digits nearby
    if re.search(r"(payment|银行卡|信用卡|customer\s*id|客户编号).{0,20}\d{4,}", text, re.I):
        matches.append("payment_or_customer_record")
        redacted = re.sub(
            r"(payment|银行卡|信用卡|customer\s*id|客户编号).{0,40}",
            r"\1 [REDACTED_PII]",
            redacted,
            flags=re.I,
        )

    return PIIResult(detected=bool(matches), redacted_text=redacted, matches=matches)


def should_reject_for_pii(text: str) -> bool:
    """Compliance rule: reject queries that themselves carry PII."""
    return detect_and_redact_pii(text).detected
