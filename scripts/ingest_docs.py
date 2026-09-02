"""Build / refresh the local vector index from data/raw (incremental-friendly)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.rag.embeddings import get_embedding_model
from app.rag.ingest import ingest_directory
from app.rag.retriever import VectorIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    raw_dir = args.raw_dir or settings.raw_dir
    index_dir = args.index_dir or settings.index_dir

    print(f"Ingesting documents from {raw_dir} ...")
    chunks = ingest_directory(
        raw_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise SystemExit("No chunks produced — check data/raw")

    print(f"Embedding {len(chunks)} chunks with {settings.embedding_model} ...")
    embedder = get_embedding_model(settings.embedding_model)

    existing_path = Path(index_dir) / "chunks.json"
    if existing_path.exists():
        existing = VectorIndex.load(index_dir)
        existing_ids = {c["chunk_id"] for c in existing.chunks}
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        if not new_chunks:
            print("No new chunks — index unchanged.")
            return
        print(f"Incremental add: {len(new_chunks)} new chunks")
        import numpy as np

        new_vecs = embedder.embed([c["text"] for c in new_chunks])
        merged_chunks = existing.chunks + new_chunks
        merged_vecs = np.vstack([existing.embeddings, new_vecs])
        index = VectorIndex(merged_chunks, merged_vecs, embedder.model_name)
    else:
        index = VectorIndex.build(chunks, embedder)

    index.save(index_dir)
    manifest = {
        "chunk_count": len(index.chunks),
        "embedding_model": index.embedding_model_name,
        "sources": sorted({c["source"] for c in index.chunks}),
    }
    (Path(index_dir) / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Index saved to {index_dir} ({manifest['chunk_count']} chunks)")
    print("Sources:", ", ".join(manifest["sources"]))


if __name__ == "__main__":
    main()
