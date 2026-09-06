from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO

import pdfplumber


@dataclass(frozen=True, slots=True)
class PageText:
    page_number: int
    text: str


class PdfTextError(RuntimeError):
    pass


def extract_pdf_pages(
    pdf_source: bytes | BinaryIO,
    *,
    max_pages: int | None = None,
) -> list[PageText]:
    source = BytesIO(pdf_source) if isinstance(pdf_source, bytes) else pdf_source
    pages: list[PageText] = []

    try:
        with pdfplumber.open(source) as pdf:
            limit = len(pdf.pages) if max_pages is None else min(max_pages, len(pdf.pages))
            for index, page in enumerate(pdf.pages[:limit], start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append(PageText(page_number=index, text=text))
    except Exception as exc:  # pdfplumber raises several parser-specific exceptions
        raise PdfTextError(f"The PDF could not be read: {exc}") from exc

    if not pages:
        raise PdfTextError(
            "No selectable text was found. This PDF may contain scanned images and need OCR first."
        )
    return pages


def chunk_pages(pages: list[PageText], *, max_characters: int = 12_000) -> list[str]:
    """Group page text into model-sized chunks without altering the document text."""
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for page in pages:
        labelled = f"[PAGE {page.page_number}]\n{page.text}"
        if current and current_size + len(labelled) > max_characters:
            chunks.append("\n\n".join(current))
            current = []
            current_size = 0

        if len(labelled) <= max_characters:
            current.append(labelled)
            current_size += len(labelled)
            continue

        # A single unusually long page is split by character count. This is only
        # transport chunking; Ollama still decides which concepts are definitions.
        start = 0
        while start < len(labelled):
            piece = labelled[start : start + max_characters]
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_size = 0
            chunks.append(piece)
            start += max_characters

    if current:
        chunks.append("\n\n".join(current))
    return chunks

