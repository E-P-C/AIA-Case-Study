"""Download a compact bilingual GGUF for embedded llama-cpp-python inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "models"

# ~1GB Q4_K_M — fits 16GB RAM alongside embedding model.
DEFAULT_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
DEFAULT_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Official Hugging Face Hub (override with HF_ENDPOINT only if you intentionally need a mirror).
DEFAULT_ENDPOINT = "https://huggingface.co"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GGUF model weights")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--filename", default=DEFAULT_FILE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT),
        help="Hugging Face Hub endpoint. Defaults to the official "
        "https://huggingface.co; set HF_ENDPOINT (e.g. to a mirror) to override.",
    )
    args = parser.parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint

    print(f"Using HF endpoint: {args.endpoint}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo}/{args.filename} ...")
    path = hf_hub_download(
        repo_id=args.repo,
        filename=args.filename,
        local_dir=str(args.outdir),
    )

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Model file not found at {target}")

    print(f"Model ready at: {target}")
    print(f"Size bytes: {target.stat().st_size}")
    print("Set MODEL_PATH in .env if you used a different filename.")

if __name__ == "__main__":
    main()
