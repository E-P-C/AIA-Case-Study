from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from app.rag.chunking import chunk_text, detect_file_type


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf_file(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to open PDF {path.name}: {exc}") from exc

    if getattr(reader, "is_encrypted", False):
        try:
            # Empty password unlocks many "owner-password only" corporate PDFs.
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"encrypted PDF not readable: {path.name}: {exc}") from exc

    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            extracted = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            extracted = ""
        pages.append(f"PAGE {i}\n{extracted}")
    return "\n\n".join(pages)


def load_document(path: Path) -> tuple[str, str]:
    """Return (text, file_type)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = load_pdf_file(path)
    else:
        text = load_text_file(path)
    file_type = detect_file_type(path.name, text)
    return text, file_type


def ingest_directory(
    raw_dir: Path,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[dict]:
    raw_dir = Path(raw_dir)
    supported = {".md", ".txt", ".pdf"}
    all_chunks: list[dict] = []
    seen_hashes: set[str] = set()

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        try:
            text, file_type = load_document(path)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {path.name}: {exc}")
            continue
        # Duplicate detection on normalized content hash
        fingerprint = " ".join(text.lower().split())[:2000]
        if fingerprint in seen_hashes:
            continue
        seen_hashes.add(fingerprint)

        for chunk in chunk_text(
            text,
            source=path.name,
            file_type=file_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ):
            all_chunks.append(chunk.to_dict())
    return all_chunks
