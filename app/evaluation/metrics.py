from __future__ import annotations

import re
from dataclasses import dataclass

CLAIM_SPLIT = re.compile(r"(?<=[。.!?;；])\s+|\n+")

# Inline citation markers (traceability metadata) are not content claims and must
# be removed before faithfulness / accuracy scoring. The generator may emit them
# in either [source:...] or (source:...) bracket form.
CITATION_RE = re.compile(r"[\[\(]source:[^\])]*[\])]")

# CJK content tokens are matched as 2-grams so small lexical variation
# (e.g. "当...时", "作答" vs "答案") does not zero out a grounded sentence.
CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
EN_WORD_RE = re.compile(r"[a-z0-9]+")

# Stopwords that carry no grounding signal.
EN_STOP = {
    "the", "and", "for", "with", "from", "this", "that", "are", "was", "were",
    "should", "must", "have", "has", "its", "can", "not", "all", "per", "may",
}

# Signals that an answer explicitly declines to answer (OOD / capability boundary).
CONTENT_REFUSAL_RE = re.compile(
    r"(cannot|cannot|can\'t|unable to|not able to|no relevant|not found|"
    r"does not contain|doesn\'t contain|no information|无法|不能|无法提供|"
    r"无相关信息|没有相关信息|找不到|未找到|知识库中无|未检索到)",
    re.IGNORECASE,
)

# Non-content framing openers that carry no factual grounding (stripped before
# faithfulness tokenisation so boilerplate does not dilute grounding overlap).
FRAMING_RE = re.compile(
    r"^(?:according to (?:the )?(?:retrieved|provided) context|"
    r"based on (?:the )?(?:retrieved|provided) context|"
    r"the retrieved context (?:states|says|indicates|mentions)|"
    r"the context provided (?:states|indicates)|"
    r"根据(?:提供|检索到|检索)的(?:信息|上下文|内容)|"
    r"基于(?:提供|检索到)的(?:信息|上下文))[，,:：]?\s*",
    re.IGNORECASE,
)


@dataclass
class MetricScores:
    faithfulness: float
    context_precision: float
    answer_accuracy: float
    details: dict


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_content_refusal(answer: str) -> bool:
    """True if the answer explicitly states it cannot find/answer from context."""
    if not answer.strip():
        return True
    return bool(CONTENT_REFUSAL_RE.search(answer))


def _cjk_bigrams(text: str) -> list[str]:
    out: list[str] = []
    for run in CJK_RUN_RE.findall(text):
        out.extend(run[i : i + 2] for i in range(len(run) - 1))
    return out


def _content_tokens(text: str) -> list[str]:
    """English words (>=3 chars, minus stopwords) + Chinese 2-grams."""
    words = [
        w for w in EN_WORD_RE.findall(text) if len(w) >= 3 and w not in EN_STOP
    ]
    return words + _cjk_bigrams(text)


def extract_key_phrases(expected: str) -> list[str]:
    """Pull numeric / short lexical anchors from the expected answer."""
    expected_n = _normalize(expected)
    phrases: list[str] = []
    # Numbers with optional units / days
    phrases.extend(re.findall(r"\d+(?:\.\d+)?(?:\s*(?:days?|day|分钟|小时|秒))?", expected_n))
    # Quoted or distinctive tokens (>=4 chars)
    for token in re.findall(r"[a-z\u4e00-\u9fff][a-z0-9\u4e00-\u9fff\-_]{3,}", expected_n):
        if token not in {"paid", "annual", "leave", "must", "shall", "with", "that", "this"}:
            phrases.append(token)
    # Keep unique, prefer longer
    uniq = []
    for p in sorted(set(phrases), key=len, reverse=True):
        if not any(p in u for u in uniq):
            uniq.append(p)
    return uniq[:8] or [expected_n[:40]]


def _phrase_in_answer(phrase: str, answer: str) -> bool:
    """Exact-substring match; CJK phrases additionally accept >=60% bigram overlap."""
    if phrase in answer:
        return True
    if CJK_RUN_RE.search(phrase):
        bigrams = _cjk_bigrams(phrase)
        if not bigrams:
            return False
        hits = sum(1 for bg in bigrams if bg in answer)
        return hits / len(bigrams) >= 0.6
    return False


def score_answer_accuracy(
    answer: str, expected: str, *, refused: bool, expect_refusal: bool
) -> float:
    if expect_refusal:
        return 1.0 if refused else 0.0
    if refused:
        return 0.0
    ans = _normalize(answer)
    keys = extract_key_phrases(expected)
    if not keys:
        return 0.0
    hits = sum(1 for k in keys if _phrase_in_answer(k, ans))
    return hits / len(keys)


def score_faithfulness(answer: str, contexts: list[str], *, refused: bool) -> float:
    """Rubric: fraction of answer sentences grounded in the retrieved context.

    Sentences are split on sentence boundaries; inline `[source:...]` citations are
    stripped (they are traceability metadata, not claims). A sentence is supported
    when >=45% of its content tokens (English words + Chinese 2-grams) appear in the
    retrieved context. Refusals without speculation score 1.0.
    """
    if refused:
        return 1.0
    ctx = _normalize(" ".join(contexts))
    if not ctx:
        return 0.0
    answer_clean = CITATION_RE.sub("", answer)
    sentences = [
        s.strip()
        for s in CLAIM_SPLIT.split(answer_clean)
        if len(s.strip()) > 2
    ]
    if not sentences:
        return 1.0 if not answer_clean.strip() else 0.0

    supported = 0
    for sent in sentences:
        if not sent:
            continue
        sent = FRAMING_RE.sub("", sent).strip()
        if not sent:
            continue
        tokens = _content_tokens(sent)
        if not tokens:
            supported += 1
            continue
        overlap = sum(1 for t in tokens if t in ctx)
        if overlap / len(tokens) >= 0.45:
            supported += 1
    return supported / len(sentences)


def score_context_precision(
    retrieved_sources: list[str],
    relevant_sources: list[str],
) -> float:
    """Context Precision = relevant retrieved chunks / retrieved chunks (source-level).

    Only meaningful when a gold relevant source exists; callers should exclude items
    with an empty gold list (OOD) from the aggregate.
    """
    if not relevant_sources:
        return 1.0
    if not retrieved_sources:
        return 0.0
    relevant_set = set(relevant_sources)
    hits = sum(1 for s in retrieved_sources if s in relevant_set)
    return hits / len(retrieved_sources)


def aggregate_metrics(rows: list[dict]) -> MetricScores:
    n = max(len(rows), 1)
    faith = sum(r["faithfulness"] for r in rows) / n
    acc = sum(r["answer_accuracy"] for r in rows) / n

    # Context Precision averaged only over items that have a gold relevant source.
    cp_rows = [r for r in rows if r.get("has_gold")]
    cprec = (
        sum(r["context_precision"] for r in cp_rows) / len(cp_rows) if cp_rows else 0.0
    )

    latencies = sorted(r["latency_ms"] for r in rows) if rows else [0]
    return MetricScores(
        faithfulness=faith,
        context_precision=cprec,
        answer_accuracy=acc,
        details={
            "n": len(rows),
            "context_precision_n": len(cp_rows),
            "latency_p90_ms": latencies[int(0.9 * (n - 1))],
            "refusal_rate": sum(1 for r in rows if r["refused"]) / n,
        },
    )
