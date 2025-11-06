"""
Data models for the document ingestion pipeline.

All models use Pydantic for validation and serialization.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document types."""

    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    TXT = "txt"
    MD = "md"
    CSV = "csv"
    JSON = "json"
    IMAGE = "image"  # png, jpg, bmp, tiff


class ProcessingStage(str, Enum):
    """Processing stages for tracking."""

    DOWNLOADED = "downloaded"
    DOC_INTEL_EXTRACTED = "doc_intel_extracted"
    IMAGES_PROCESSED = "images_processed"
    CONTENT_ASSEMBLED = "content_assembled"
    CHUNKED = "chunked"
    SUMMARY_GENERATED = "summary_generated"
    METADATA_ENRICHED = "metadata_enriched"
    EMBEDDINGS_GENERATED = "embeddings_generated"
    INDEXED = "indexed"
    FAILED = "failed"


class ImageInfo(BaseModel):
    """Information about an extracted image."""

    image_id: str
    page_number: int | None = None
    position: dict[str, float] | None = None  # {x, y, width, height}
    ocr_text: str = ""
    description: str = ""
    image_type: str = ""  # chart, diagram, photo, etc.
    confidence: float = 0.0


class PageContent(BaseModel):
    """Content from a single page."""

    page_number: int
    content: str
    images: list[ImageInfo] = Field(default_factory=list)
    has_tables: bool = False
    table_count: int = 0


class DocumentMetadata(BaseModel):
    """Metadata enriched from CSV lookup."""

    file_name: str
    document_id: str | None = None
    library_name_en: str = "Unknown"
    library_name_tc: str = "未知"
    category_name_en: str = "Uncategorized"
    category_name_tc: str = "未分類"
    title_name_en: str = ""
    title_name_tc: str = ""
    item_url: str = ""
    has_images: bool = False
    image_count: int = 0
    # Additional CSV fields stored as key-value
    additional_fields: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON string for metadata_file field."""
        return self.model_dump_json(exclude={"additional_fields"})


class ChunkMetadata(BaseModel):
    """Metadata for a single chunk."""

    page_number: int | None = None
    chunk_section: str = ""
    has_images: bool = False
    image_count: int = 0
    image_types: list[str] = Field(default_factory=list)
    image_page_numbers: list[int] = Field(default_factory=list)
    # CSV enrichment fields
    document_id: str | None = None
    library_name_en: str = ""
    library_name_tc: str = ""
    category_name_en: str = ""
    category_name_tc: str = ""
    # Additional CSV fields
    additional_fields: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON string for metadata_chunk field."""
        return self.model_dump_json()


class DocumentChunk(BaseModel):
    """A single chunk of document content."""

    chunk_id: str
    content: str
    page_number: int | None = None
    chunk_index: int
    metadata: ChunkMetadata


class EnrichedDocument(BaseModel):
    """Document after Document Intelligence and image processing."""

    filename: str
    title: str = ""
    total_pages: int
    pages: list[PageContent]
    markdown: str
    images: list[ImageInfo] = Field(default_factory=list)
    doc_type: DocumentType
    processing_metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentSummary(BaseModel):
    """Generated summary for a document."""

    purpose: str
    page_summaries: dict[int, list[str]]  # page_number -> list of key sentences
    full_summary: str
    total_pages: int
    generation_method: str = "extractive"


class SearchDocument(BaseModel):
    """
    Document structure for Azure AI Search indexing.

    This matches the exact schema specified in the requirements:
    - id, filename, file_summary, file_summary_chunk (vector),
    - metadata_file (JSON), content_chunk, metadata_chunk (JSON),
    - content_chunk_dim (vector)
    """

    id: str = Field(..., description="Unique chunk ID")
    filename: str = Field(..., description="Source document filename")
    file_summary: str = Field(..., description="Generated document summary")
    file_summary_chunk: list[float] = Field(
        ..., description="Vector embedding of file summary (1536 dims)"
    )
    metadata_file: str = Field(..., description="JSON string with document metadata")
    content_chunk: str = Field(..., description="Chunk text content")
    metadata_chunk: str = Field(..., description="JSON string with chunk metadata")
    content_chunk_dim: list[float] = Field(
        ..., description="Vector embedding of content chunk (1536 dims)"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Azure Search upload."""
        return self.model_dump()


class ProcessingResult(BaseModel):
    """Result of processing a single document."""

    filename: str
    success: bool
    stage: ProcessingStage
    chunks_created: int = 0
    images_processed: int = 0
    error_message: str | None = None
    processing_time_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchProcessingResult(BaseModel):
    """Result of processing a batch of documents."""

    total_documents: int
    successful: int
    failed: int
    total_chunks: int
    total_images: int
    results: list[ProcessingResult]
    total_time_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.successful / self.total_documents) * 100


class BlobFile(BaseModel):
    """Information about a file in Azure Blob Storage."""

    name: str
    size: int
    last_modified: datetime
    content_type: str | None = None

    @property
    def extension(self) -> str:
        """Get file extension."""
        return Path(self.name).suffix.lower().lstrip(".")

    @property
    def document_type(self) -> DocumentType:
        """Determine document type from extension."""
        ext = self.extension
        if ext == "pdf":
            return DocumentType.PDF
        elif ext in ["docx", "doc"]:
            return DocumentType.DOCX
        elif ext in ["xlsx", "xls"]:
            return DocumentType.XLSX
        elif ext in ["pptx", "ppt"]:
            return DocumentType.PPTX
        elif ext == "txt":
            return DocumentType.TXT
        elif ext == "md":
            return DocumentType.MD
        elif ext == "csv":
            return DocumentType.CSV
        elif ext == "json":
            return DocumentType.JSON
        elif ext in ["png", "jpg", "jpeg", "bmp", "tiff", "tif"]:
            return DocumentType.IMAGE
        else:
            return DocumentType.TXT  # Default


# Helper functions


def generate_chunk_id(document_id: str, page_number: int | None, chunk_index: int) -> str:
    """
    Generate unique chunk ID.

    Format: {document_id}_p{page}_c{chunk}
    Example: DOC001_p5_c3 (document DOC001, page 5, chunk 3)
    """
    page_part = f"p{page_number}" if page_number is not None else "p0"
    return f"{document_id}_{page_part}_c{chunk_index}"


def parse_chunk_id(chunk_id: str) -> tuple[str, int | None, int]:
    """
    Parse chunk ID back to components.

    Robustly handles document IDs that may contain underscores by
    taking the last two underscore-separated parts as page and chunk.

    Returns: (document_id, page_number, chunk_index)
    """
    parts = chunk_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid chunk ID format: {chunk_id}")

    # Support filenames/stems that contain underscores by consuming from the end
    page_part = parts[-2]
    chunk_part = parts[-1]
    document_id = "_".join(parts[:-2])

    page_str = page_part.lstrip("p")
    chunk_str = chunk_part.lstrip("c")

    page_number = int(page_str) if page_str != "0" else None
    chunk_index = int(chunk_str)

    return document_id, page_number, chunk_index
