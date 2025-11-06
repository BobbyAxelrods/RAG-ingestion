import logging
from typing import List

try:
    # Prefer langchain's splitters if available
    from langchain_text_splitters import (
        MarkdownTextSplitter,
        RecursiveCharacterTextSplitter,
    )
except Exception:
    MarkdownTextSplitter = None  # type: ignore
    RecursiveCharacterTextSplitter = None  # type: ignore

from src.etl.config import ChunkingConfig
from src.etl.models.document_models import EnrichedDocument


logger = logging.getLogger(__name__)


class DocAnalysisChunker:
    """Chunker that operates on EnrichedDocument produced by Document Analysis.

    Default behavior is page-based chunking: one chunk per page, with an
    overlap of configurable characters from the previous page (default 500).
    If page content is unavailable, falls back to splitting the assembled
    markdown using configured sizes.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config
        # For fallback text splitting
        self.max_chunk_size = max(int(getattr(config, "chunk_size", 1000) or 1000), 1)
        # Interpret chunk_overlap as character overlap between pages when using page-based chunking
        self.page_overlap_chars = max(int(getattr(config, "chunk_overlap", 500) or 500), 0)
        # Backward compatibility for fallback splitters
        self.token_overlap = self.page_overlap_chars

    def _simple_split(self, text: str) -> List[str]:
        """Simple character-based splitter with overlap as a fallback."""
        chunks: List[str] = []
        size = self.max_chunk_size
        overlap = min(self.token_overlap, size - 1) if size > 1 else 0
        start = 0
        n = len(text)
        while start < n:
            end = min(start + size, n)
            chunks.append(text[start:end])
            if end >= n:
                break
            start = max(end - overlap, start + 1)
        return chunks

    def _split_text(self, text: str) -> List[str]:
        if not text:
            return []

        # Try Markdown splitter first (better for structured text)
        if MarkdownTextSplitter is not None:
            try:
                splitter = MarkdownTextSplitter(
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=self.token_overlap,
                )
                return splitter.split_text(text)
            except Exception as e:
                logger.debug("Markdown splitter failed, falling back: %s", e)

        # Fallback to recursive character splitter if available
        if RecursiveCharacterTextSplitter is not None:
            try:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=self.token_overlap,
                    add_start_index=False,
                )
                return splitter.split_text(text)
            except Exception as e:
                logger.debug("Recursive splitter failed, falling back: %s", e)

        # Final fallback: simple character-based splitting
        return self._simple_split(text)

    def chunk_document(self, enriched: EnrichedDocument) -> List[dict]:
        """Create chunk dicts that DocumentProcessor expects.

        Page-based chunking: one chunk per page, each including an overlap
        of `page_overlap_chars` from the previous page's tail to preserve
        cross-page context. If no per-page content is available, fall back
        to markdown-based splitting.
        """
        chunks: List[dict] = []

        pages = getattr(enriched, "pages", []) or []
        if pages:
            total = len(pages)
            prev_tail = ""
            for i, page in enumerate(pages, start=1):
                page_text = page.content or ""
                # Compose chunk with overlap from previous page
                prefix = prev_tail if prev_tail else ""
                chunk_text = f"{prefix}{page_text}" if prefix else page_text

                # Prepare simple image metadata for the page
                image_count = len(getattr(page, "images", []) or [])
                image_types = [img.image_type for img in (getattr(page, "images", []) or []) if getattr(img, "image_type", "")]

                chunk_meta = {
                    "chunk_id": page.page_number,
                    "chunk_index": page.page_number,
                    "filename": enriched.filename,
                    "title": enriched.title,
                    "total_pages": enriched.total_pages,
                    "total_chunks": total,
                    "page_number": page.page_number,
                    "page_numbers": [page.page_number],
                    "has_images": image_count > 0,
                    "image_count": image_count,
                    "image_types": image_types,
                }

                chunks.append({
                    "content": chunk_text,
                    "metadata": chunk_meta,
                })

                # Update tail for next page's overlap
                if self.page_overlap_chars > 0:
                    prev_tail = page_text[-self.page_overlap_chars :] if page_text else ""
                else:
                    prev_tail = ""

            logger.info(
                "DocAnalysisChunker (page-based) produced %d chunks for %s with %d-char overlap",
                len(chunks), enriched.filename, self.page_overlap_chars,
            )
            return chunks

        # Fallback: split assembled markdown if page content is missing
        markdown = enriched.markdown or ""
        raw_chunks = self._split_text(markdown)
        for idx, chunk_text in enumerate(raw_chunks, start=1):
            chunk_meta = {
                "chunk_id": idx,
                "chunk_index": idx,
                "filename": enriched.filename,
                "title": enriched.title,
                "total_pages": enriched.total_pages,
                "total_chunks": len(raw_chunks),
                # page_number left None in fallback
            }
            chunks.append({
                "content": chunk_text,
                "metadata": chunk_meta,
            })

        logger.info("DocAnalysisChunker (fallback) produced %d chunks for %s", len(chunks), enriched.filename)
        return chunks

