"""Run offline evaluation against eval_set.json and emit metric tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.metrics import (
    aggregate_metrics,
    is_content_refusal,
    score_answer_accuracy,
    score_context_precision,
    score_faithfulness,
)
from app.rag.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=ROOT / "app" / "evaluation" / "eval_set.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "eval_results.json",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--reranker", choices=["on", "off", "default"], default="default")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip model warm-up (keeps cold model-load in item-1 latency)",
    )
    args = parser.parse_args()

    items = json.loads(args.eval_set.read_text(encoding="utf-8"))
    pipeline = RAGPipeline()

    rerank = None
    if args.reranker == "on":
        rerank = True
    elif args.reranker == "off":
        rerank = False

    # Steady-state SLO: the SLO applies to a running service where the embedded
    # model is already loaded. Warm the model outside the timed loop so item-1
    # latency does not include cold-start (documented deployment trade-off).
    if not args.no_warmup:
        gen = pipeline.load_generator()
        gen.generate(
            question="warmup",
            contexts=["AKP internal knowledge base warm-up."],
            history=None,
            temperature=0.2,
            max_tokens=32,
        )
        print("Model warmed (cold load excluded from measured latencies).")

    rows = []
    for item in items:
        result = pipeline.ask(
            item["question"],
            session_id=None,
            top_k=args.top_k,
            reranker_enabled=rerank,
            temperature=args.temperature,
        )
        contexts = [c.get("snippet", "") for c in result.retrieved_chunks]
        retrieved_sources = [c.get("source", "") for c in result.retrieved_chunks]
        expect_refusal = bool(item.get("expect_refusal") or item.get("expect_injection_block"))
        relevant = item.get("relevant_sources", [])
        has_gold = bool(relevant)

        # A refusal is counted when the pipeline gates it OR the model explicitly
        # states it cannot find the information (OOD capability-boundary handling).
        effectively_refused = result.refused or (
            expect_refusal and is_content_refusal(result.answer)
        )

        faith = score_faithfulness(
            result.answer, contexts, refused=effectively_refused
        )
        cprec = score_context_precision(retrieved_sources, relevant)
        acc = score_answer_accuracy(
            result.answer,
            item["expected_answer"],
            refused=effectively_refused,
            expect_refusal=expect_refusal,
        )
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": result.answer,
                "refused": result.refused,
                "effectively_refused": effectively_refused,
                "refusal_reason": result.refusal_reason,
                "has_gold": has_gold,
                "faithfulness": round(faith, 4),
                "context_precision": round(cprec, 4),
                "answer_accuracy": round(acc, 4),
                "latency_ms": result.latency_ms,
                "token_in": result.token_in,
                "token_out": result.token_out,
                "citations": result.citations,
            }
        )
        print(
            f"[{item['id']}] acc={acc:.2f} faith={faith:.2f} cprec={cprec:.2f} "
            f"latency={result.latency_ms}ms refused={result.refused}"
        )

    summary = aggregate_metrics(rows)
    payload = {
        "summary": {
            "faithfulness": round(summary.faithfulness, 4),
            "context_precision": round(summary.context_precision, 4),
            "answer_accuracy": round(summary.answer_accuracy, 4),
            **summary.details,
            "targets": {
                "faithfulness": 0.85,
                "context_precision": 0.70,
                "answer_accuracy": 0.80,
                "latency_p90_ms": 10000,
            },
            "pass": {
                "faithfulness": summary.faithfulness >= 0.85,
                "context_precision": summary.context_precision >= 0.70,
                "answer_accuracy": summary.answer_accuracy >= 0.80,
                "latency_p90": summary.details["latency_p90_ms"] <= 10000,
            },
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
