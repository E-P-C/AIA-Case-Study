# AKP Internal RAG QA Service

Junior Backend case-study demo: bilingual (CN/EN) retrieval-augmented QA over the AKP internal knowledge corpus, with citations, multi-turn sessions, PII / prompt-injection guards, and structured logging.

## Design choice: embedded `llama-cpp-python`

This demo loads a **GGUF model inside the Python process** via `llama-cpp-python`. **No** LM Studio / `llama-server` / other external inference server is required.

**Trade-offs (intentional for clone-and-run):**
- Every process restart reloads a multi‑GB GGUF from disk → repeated heavy I/O and cold-start latency.
- The model does **not** auto-unload on idle; RAM is released only when the Python process exits.
- Suitable for a course-project demonstration. For production, migrate to a standalone inference server with JIT load + TTL idle-unload to reduce disk wear and optimize memory.

**Hardware baseline:** 16GB RAM + SSD.

Default model: `Qwen2.5-1.5B-Instruct` Q4_K_M (~1GB) — bilingual, fits beside the embedding model on 16GB machines.

## Quick start

```bash
# from RAGagent/
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
# Windows note: if llama-cpp-python tries to compile from source, use:
#   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
copy .env.example .env   # or: cp .env.example .env

# Downloads from Hugging Face mirror (or https://huggingface.co), ~1.1GB
$env:HF_ENDPOINT = "https://hf-mirror.com" # Ignore this line ONLY IF you can reach HF official site
python scripts/download_model.py
python scripts/ingest_docs.py
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Chat UI
Open **http://127.0.0.1:8000/chat** — a lightweight browser chat box backed by `POST /v1/ask` (bubbles, citations, multi-turn in the same session, API token pre-filled).

## Project layout

```
RAGagent/
  app/                 # FastAPI + RAG pipeline (+ app/static/chat.html UI)
  data/raw/            # Knowledge corpus (md/txt/pdf)
  data/index/          # Built vector index (generated)
  models/              # GGUF weights (downloaded)
  scripts/             # download / ingest / eval / sensitivity
  logs/                # JSONL request + security logs
  docs/                # Design note + evaluation summary
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `TOP_K` | `4` | Spec baseline retrieval depth |
| `SIMILARITY_THRESHOLD` | `0.45` | Below → explanatory refusal |
| `CONTEXT_GATE_RATIO` | `0.80` | Keep chunks scoring ≥ 80% of the best hit (context pruning) |
| `RERANKER_ENABLED` | `true` | Lexical reranker on/off |
| `TEMPERATURE` | `0.2` | Generation temperature (0.1–0.9) |
| `REPEAT_PENALTY` | `1.3` | Suppresses repetition loops in generation |
| `SESSION_IDLE_TIMEOUT_SEC` | `1800` | 30‑minute idle session purge |
| `API_TOKEN` | `akp-demo-token` | Simple token auth header |

## Evaluation & cost sensitivity

```bash
python scripts/run_eval.py
python scripts/sensitivity_analysis.py
```

- Metrics: answer accuracy, Faithfulness, Context Precision (rubrics in `app/evaluation/metrics.py`)
- Results: `docs/eval_results.json` (per-item + summary), `docs/sensitivity.json` (cost/sensitivity)
- Deliverable write-ups: `docs/DESIGN_NOTE.md`, `docs/EVALUATION.md`
- PII-redacted sample logs: `logs/sample_requests.jsonl`, `logs/sample_security_audit.jsonl`

The eval scripts warm the embedded model before the timed loop, so the reported
P90 reflects steady-state serving; cold model load (~15s, documented trade-off) is
excluded as a startup cost, not per-query latency.

## API sketch

`POST /v1/ask`

```json
{
  "question": "How many remote days per week are allowed?",
  "session_id": "optional-uuid",
  "top_k": 4,
  "reranker_enabled": true,
  "temperature": 0.2
}
```

Response includes `answer`, `citations[]`, `retrieved_chunks[]`, token usage, `latency_ms`, and refusal metadata when applicable.

`POST /v1/ask` requires the `X-API-Token` header (default `akp-demo-token`); `/docs` exposes it via the **Authorize** button. `GET /v1/health` is public.

## Notes on corpus

- `architecture_doc_scanned.txt` simulates OCR output (intentional misspellings cleaned during ingest).
- Native PDFs under `data/raw/` are text-extracted with `pypdf`.
- Indexing supports **incremental** adds without full rebuild when new files appear.

## Developer functions
Health check:

```bash
curl http://127.0.0.1:8000/v1/health
```

Ask (API token from `.env`, default `akp-demo-token`):

```bash
curl -X POST http://127.0.0.1:8000/v1/ask ^
  -H "Content-Type: application/json" ^
  -H "X-API-Token: akp-demo-token" ^
  -d "{\"question\": \"What is the default top_k?\", \"session_id\": \"demo-1\"}"
```


Developer console (`/docs`, Swagger UI): for developers who want to inspect each endpoint.
Click **Authorize** in the top-right corner, enter the API token (`akp-demo-token`), then
expand **POST /v1/ask**, click **Try it out**, fill in `question` (and an optional
`session_id`) and execute. Reusing the same `session_id` keeps multi-turn continuity.

CLI (no HTTP):

```bash
python scripts/cli_demo.py --question "病假连续超过几天需要医疗证明？"
```
