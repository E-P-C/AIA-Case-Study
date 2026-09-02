from __future__ import annotations

import re
from dataclasses import asdict, dataclass


OCR_FIXES = {
    "Syste,m": "System",
    "Architcture": "Architecture",
    "documments": "documents",
    "extrct": "extract",
    "moudle": "module",
}


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    file_type: str
    page: int
    language: str = "mixed"

    def to_dict(self) -> dict:
        return asdict(self)


def clean_ocr_noise(text: str) -> str:
    cleaned = text
    for bad, good in OCR_FIXES.items():
        cleaned = cleaned.replace(bad, good)
    # Drop repeated page-footer noise
    cleaned = re.sub(
        r"\[Page footer\].*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def detect_file_type(path_name: str, text: str) -> str:
    lower = path_name.lower()
    if "scanned" in lower or "OCR EXTRACTED" in text[:200]:
        return "scanned-pdf"
    if lower.endswith(".pdf"):
        return "native-pdf"
    if lower.endswith(".docx"):
        return "docx"
    return "plain-text"


def infer_pages(text: str) -> list[tuple[int, str]]:
    """Split text into (page_number, page_text) using PAGE markers when present."""
    page_splits = list(re.finditer(r"(?im)^PAGE\s+(\d+)\b.*$", text))
    if not page_splits:
        # Markdown / plain docs: treat ## headings as soft page proxies (1-indexed sections)
        sections = re.split(r"(?m)(?=^#{1,3}\s)", text)
        sections = [s.strip() for s in sections if s.strip()]
        if len(sections) <= 1:
            return [(1, text)]
        return [(i + 1, sec) for i, sec in enumerate(sections)]

    pages: list[tuple[int, str]] = []
    for i, match in enumerate(page_splits):
        start = match.end()
        end = page_splits[i + 1].start() if i + 1 < len(page_splits) else len(text)
        page_no = int(match.group(1))
        pages.append((page_no, text[start:end].strip()))
    return pages or [(1, text)]


def chunk_text(
    text: str,
    *,
    source: str,
    file_type: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[Chunk]:
    text = clean_ocr_noise(text)
    chunks: list[Chunk] = []
    counter = 0
    for page_no, page_text in infer_pages(text):
        if not page_text:
            continue
        start = 0
        while start < len(page_text):
            end = min(start + chunk_size, len(page_text))
            # Prefer breaking on paragraph/sentence boundaries
            if end < len(page_text):
                window = page_text[start:end]
                break_at = max(window.rfind("\n\n"), window.rfind("。"), window.rfind(". "))
                if break_at > chunk_size * 0.4:
                    end = start + break_at + 1
            piece = page_text[start:end].strip()
            if piece:
                counter += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{source}::p{page_no}::c{counter}",
                        text=piece,
                        source=source,
                        file_type=file_type,
                        page=page_no,
                    )
                )
            if end >= len(page_text):
                break
            start = max(0, end - chunk_overlap)
    return chunks
