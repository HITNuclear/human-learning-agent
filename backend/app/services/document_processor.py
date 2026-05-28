from __future__ import annotations
import io
import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader
from app.config import settings
from app.utils.text_chunker import chunk_text


async def process_pdf(content: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(content))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(text.strip())
    full_text = "\n\n".join(pages_text)
    return chunk_text(
        full_text,
        chunk_size=settings.chunk_token_size,
        overlap=settings.chunk_token_overlap,
        tokenizer_name=settings.embedding_model,
    )


def process_text(text: str) -> list[str]:
    return chunk_text(
        text,
        chunk_size=settings.chunk_token_size,
        overlap=settings.chunk_token_overlap,
        tokenizer_name=settings.embedding_model,
    )


async def process_url(url: str) -> list[str]:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script/style elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Extract meaningful text blocks
    blocks: list[str] = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote"]):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            blocks.append(text)

    full_text = "\n\n".join(blocks)
    return chunk_text(
        full_text,
        chunk_size=settings.chunk_token_size,
        overlap=settings.chunk_token_overlap,
        tokenizer_name=settings.embedding_model,
    )
