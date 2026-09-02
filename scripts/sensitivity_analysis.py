"""Token-cost estimate per 1,000 calls + sensitivity for top_k / reranker / temperature."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.rag.pipeline import RAGPipeline

SAMPLE_QUESTIONS = [
    "What is the default top_k for retrieval?",
    "混合办公每周最多几天远程？",
    "What happens when similarity is below 0.45?",
    "How should scanned PDFs be handled during indexing?",
    "Can employees use public ChatGPT with confidential data?",
]


def estimate_cost(token_in: float, token_out: float, settings) -> float:
    return (
        (token_in / 1_000_000.0) * settings.cost_per_1m_input_tokens
        + (token_out / 1_000_000.0) * settings.cost_per_1m_output_tokens
    )


def run_config(pipeline: RAGPipeline, *, top_k: int, rerank: bool, temperature: float) -> dict:
    latencies = []
    tin = []
    tout = []
    refused = 0
    for q in SAMPLE_QUESTIONS:
        r = pipeline.ask(
            q,
            session_id=None,
            top_k=top_k,
            reranker_enabled=rerank,
            temperature=temperature,
        )
        latencies.append(r.latency_ms)
        tin.append(r.token_in)
        tout.append(r.token_out)
        refused += int(r.refused)
    settings = get_settings()
    avg_in = statistics.mean(tin) if tin else 0
    avg_out = statistics.mean(tout) if tout else 0
    per_call = estimate_cost(avg_in, avg_out, settings)
    return {
        "top_k": top_k,
        "reranker": rerank,
        "temperature": temperature,
        "avg_token_in": round(avg_in, 1),
        "avg_token_out": round(avg_out, 1),
        "avg_latency_ms": int(statistics.mean(latencies)),
        "p90_latency_ms": sorted(latencies)[int(0.9 * (len(latencies) - 1))],
        "refusal_rate": refused / len(SAMPLE_QUESTIONS),
        "est_cost_per_call_usd": round(per_call, 6),
        "est_cost_per_1000_calls_usd": round(per_call * 1000, 4),
        "quality_note": (
            "lower temperature → more deterministic grounded answers; "
            "higher top_k → more context tokens / cost (context-gate caps the effect); "
            "reranker ON adds CPU latency and on this corpus can favor keyword-heavy "
            "overlapping PDFs — see docs/EVALUATION.md for measured quality per config"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "sensitivity.json",
    )
    args = parser.parse_args()

    pipeline = RAGPipeline()
    settings = get_settings()

    # Warm the embedded model so measured latencies reflect steady-state
    # serving (cold model-load is a startup cost, not per-query latency).
    gen = pipeline.load_generator()
    gen.generate(
        question="warmup",
        contexts=["AKP internal knowledge base warm-up."],
        history=None,
        temperature=0.2,
        max_tokens=32,
    )
    print("Model warmed (cold load excluded from measured latencies).")

    configs = []
    # Dimension 1: top_k
    for k in (2, 4, 6):
        configs.append(run_config(pipeline, top_k=k, rerank=True, temperature=0.2))
    # Dimension 2: reranker on/off at baseline top_k
    configs.append(run_config(pipeline, top_k=4, rerank=False, temperature=0.2))
    # Dimension 3: temperature
    for temp in (0.1, 0.4, 0.7):
        configs.append(run_config(pipeline, top_k=4, rerank=True, temperature=temp))

    # Deduplicate identical rows from overlapping baseline
    unique = []
    seen = set()
    for row in configs:
        key = (row["top_k"], row["reranker"], row["temperature"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    baseline = next(
        r for r in unique if r["top_k"] == 4 and r["reranker"] is True and r["temperature"] == 0.2
    )
    payload = {
        "assumptions": {
            "cost_per_1m_input_tokens_usd": settings.cost_per_1m_input_tokens,
            "cost_per_1m_output_tokens_usd": settings.cost_per_1m_output_tokens,
            "note": (
                "Embedded GGUF inference has near-zero API token cost; figures are "
                "equivalent cloud-token costs for budgeting / sensitivity comparison."
            ),
            "sample_questions": SAMPLE_QUESTIONS,
        },
        "baseline_per_1000_calls_usd": baseline["est_cost_per_1000_calls_usd"],
        "rows": unique,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
