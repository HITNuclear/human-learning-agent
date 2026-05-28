from __future__ import annotations

from functools import lru_cache
from transformers import AutoTokenizer


@lru_cache(maxsize=4)
def _get_tokenizer(tokenizer_name: str):
    return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def _token_count(text: str, tokenizer) -> int:
    if not text:
        return 0
    ids = tokenizer(text, add_special_tokens=False, truncation=False).get("input_ids") or []
    return len(ids)


def _tail_by_tokens(text: str, tokenizer, token_overlap: int) -> str:
    if not text or token_overlap <= 0:
        return ""
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping") or []
    if not offsets:
        return ""
    if token_overlap >= len(offsets):
        return text.strip()
    start_char = offsets[-token_overlap][0]
    return text[start_char:].strip()


def _split_long_paragraph(para: str, tokenizer, chunk_size: int, overlap: int) -> list[str]:
    encoded = tokenizer(
        para,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping") or []
    if not offsets:
        return []

    step = max(1, chunk_size - overlap)
    windows: list[str] = []
    start = 0
    total = len(offsets)
    while start < total:
        end = min(start + chunk_size, total)
        start_char = offsets[start][0]
        end_char = offsets[end - 1][1]
        piece = para[start_char:end_char].strip()
        if piece:
            windows.append(piece)
        if end >= total:
            break
        start += step
    return windows


def chunk_text(
    text: str,
    chunk_size: int = 384,
    overlap: int = 64,
    tokenizer_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> list[str]:
    """Token-aware chunking with paragraph-first packing and fixed token overlap."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunk_size = max(8, int(chunk_size))
    overlap = max(0, int(overlap))
    if overlap >= chunk_size:
        overlap = max(1, chunk_size // 4)

    tokenizer = _get_tokenizer(tokenizer_name)

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para_tokens = _token_count(para, tokenizer)

        if para_tokens > chunk_size:
            if current.strip():
                chunks.append(current.strip())
                current = _tail_by_tokens(current, tokenizer, overlap)

            long_parts = _split_long_paragraph(para, tokenizer, chunk_size, overlap)
            chunks.extend(long_parts)
            current = _tail_by_tokens(long_parts[-1], tokenizer, overlap) if long_parts else ""
            continue

        if not current:
            current = para
            continue

        candidate = f"{current}\n\n{para}"
        if _token_count(candidate, tokenizer) <= chunk_size:
            current = candidate
            continue

        chunks.append(current.strip())
        carry = _tail_by_tokens(current, tokenizer, overlap)
        current = f"{carry}\n\n{para}".strip() if carry else para

        if _token_count(current, tokenizer) > chunk_size:
            long_parts = _split_long_paragraph(current, tokenizer, chunk_size, overlap)
            if long_parts:
                chunks.extend(long_parts[:-1])
                current = long_parts[-1]

    if current.strip():
        chunks.append(current.strip())

    deduped: list[str] = []
    for chunk in chunks:
        if chunk and (not deduped or deduped[-1] != chunk):
            deduped.append(chunk)

    normalized: list[str] = []
    for chunk in deduped:
        if _token_count(chunk, tokenizer) <= chunk_size:
            normalized.append(chunk)
            continue
        normalized.extend(_split_long_paragraph(chunk, tokenizer, chunk_size, overlap))

    final_chunks: list[str] = []
    for chunk in normalized:
        if chunk and (not final_chunks or final_chunks[-1] != chunk):
            final_chunks.append(chunk)
    return final_chunks
