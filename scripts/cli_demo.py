"""Interactive CLI demo for local smoke testing without HTTP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--session-id", type=str, default="cli-demo")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    print("AKP RAG QA CLI — type 'exit' to quit. Multi-turn uses the same session_id.\n")

    session_id = args.session_id
    if args.question:
        questions = [args.question]
    else:
        questions = None

    while True:
        if questions is not None:
            if not questions:
                break
            q = questions.pop(0)
            print(f"> {q}")
        else:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q or q.lower() in {"exit", "quit"}:
                break

        result = pipeline.ask(q, session_id=session_id)
        session_id = result.session_id
        print(f"\n[{result.request_id}] latency={result.latency_ms}ms refused={result.refused}")
        print(result.answer)
        if result.citations:
            print("Citations:", result.citations)
        print()


if __name__ == "__main__":
    main()
