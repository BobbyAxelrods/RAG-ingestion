import logging
from typing import List

try:
    # Prefer langchain's splitters if available
    from langchain_text_splitters import (
        MarkdownTextSplitter,
        RecursiveCharacterTextSplitter,
        PythonCodeTextSplitter,
    )
except Exception:
    MarkdownTextSplitter = None  # type: ignore
    RecursiveCharacterTextSplitter = None  # type: ignore
    PythonCodeTextSplitter = None  # type: ignore

from src.etl.config import ChunkingConfig
from src.etl.models.document_models import EnrichedDocument


logger = logging.getLogger(__name__)


class LangChainChunker:
    """Chunker that splits `EnrichedDocument.markdown` using langchain splitters or
    a safe fallback, matching DocumentProcessor expectations.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.max_chunk_size = max(int(getattr(config, "chunk_size", 1000) or 1000), 1)
        self.token_overlap = max(int(getattr(config, "chunk_overlap", 100) or 100), 0)

    def _simple_split(self, text: str) -> List[str]:
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

    def _split_text(self, text: str, doc_type: str | None) -> List[str]:
        if not text:
            return []

        # Prefer format-aware splitters when available
        if doc_type == "python" and PythonCodeTextSplitter is not None:
            try:
                splitter = PythonCodeTextSplitter(
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=self.token_overlap,
                )
                return splitter.split_text(text)
            except Exception as e:
                logger.debug("Python splitter failed, fallback: %s", e)

        if MarkdownTextSplitter is not None:
            try:
                splitter = MarkdownTextSplitter(
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=self.token_overlap,
                )
                return splitter.split_text(text)
            except Exception as e:
                logger.debug("Markdown splitter failed, fallback: %s", e)

        if RecursiveCharacterTextSplitter is not None:
            try:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=self.token_overlap,
                    add_start_index=False,
                )
                return splitter.split_text(text)
            except Exception as e:
                logger.debug("Recursive splitter failed, fallback: %s", e)

        return self._simple_split(text)

    def chunk_document(self, enriched: EnrichedDocument) -> List[dict]:
        markdown = enriched.markdown or ""
        raw_chunks = self._split_text(markdown, getattr(enriched, "doc_type", None))

        chunks: List[dict] = []
        for idx, chunk_text in enumerate(raw_chunks, start=1):
            chunk_meta = {
                "chunk_id": idx,
                "filename": enriched.filename,
                "title": enriched.title,
                "total_pages": enriched.total_pages,
            }
            chunks.append({
                "content": chunk_text,
                "metadata": chunk_meta,
            })

        logger.info("LangChainChunker produced %d chunks for %s", len(chunks), enriched.filename)
        return chunks

