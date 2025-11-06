"""
ETL output models aligned with project_definition/etl_schema.json.

These Pydantic models mirror the target JSON structure produced by the
indexing pipeline, ensuring schema-consistent outputs for downstream systems
like Azure AI Search and validation tooling.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0


class ChunkMetadata(BaseModel):
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    bbox: Optional[BBox] = None


class Chunk(BaseModel):
    chunk_id: str
    chunk_text: str
    chunk_position: int
    chunk_type: str = "text"
    chunk_semantic_density: float = 0.0
    chunk_metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    content_chunk_vector: List[float] = Field(default_factory=list)


class PageSummary(BaseModel):
    page_function_summary: str = ""
    page_summary_vector: List[float] = Field(default_factory=list)


class PageMetadata(BaseModel):
    page_width: float = 0.0
    page_height: float = 0.0
    page_orientation: str = ""
    has_images: bool = False
    has_tables: bool = False
    image_count: int = 0
    table_count: int = 0


class Page(BaseModel):
    page_number: int
    page_summary: PageSummary = Field(default_factory=PageSummary)
    page_metadata: PageMetadata = Field(default_factory=PageMetadata)
    chunks: List[Chunk] = Field(default_factory=list)
    # Per-page insights
    page_keyword_extraction: KeywordExtraction = Field(default_factory=lambda: KeywordExtraction())
    page_qna_pairs: List[QAPair] = Field(default_factory=list)


class FileSummary(BaseModel):
    file_function_summary: str = ""
    file_summary_vector: List[float] = Field(default_factory=list)


class FileMetadata(BaseModel):
    file_name: str
    file_path: str = ""
    file_size_bytes: int = 0
    file_type: str = ""
    library_name: str = ""
    category_name: str = ""
    title_name: str = ""
    file_url: str = ""
    branch_name: str = ""
    document_language: str = ""
    last_updated: Optional[datetime] = None
    page_count: int = 0
    extracted_at: Optional[datetime] = None
    processing_version: str = ""
    # Additional CSV columns mapped by their original column names
    additional_fields: dict[str, Optional[str]] = Field(default_factory=dict)


class KeywordExtraction(BaseModel):
    # Base fields for backward compatibility
    entities: List[str] = Field(default_factory=list)
    product_names: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    file_type: str = ""
    # Page-level key phrases removed per requirements; use None to omit on serialization
    key_phrases: Optional[List[str]] = None
    # Bilingual specificity
    entities_en: List[str] = Field(default_factory=list)
    entities_tc: List[str] = Field(default_factory=list)
    product_names_en: List[str] = Field(default_factory=list)
    product_names_tc: List[str] = Field(default_factory=list)
    topics_en: List[str] = Field(default_factory=list)
    topics_tc: List[str] = Field(default_factory=list)


class QAPair(BaseModel):
    question: str
    answer: str
    qa_doc_id: str
    qa_page_number: int
    qa_chunk_ids: List[str] = Field(default_factory=list)
    qa_confidence: float = 0.0


class Table(BaseModel):
    table_id: str
    page_number: int
    table_position: int
    table_html: str = ""
    table_markdown: str = ""
    table_csv: str = ""
    table_caption: str = ""
    table_summary: str = ""
    column_headers: List[str] = Field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    bbox: Optional[BBox] = None


class Image(BaseModel):
    image_id: str
    page_number: int
    image_position: int
    image_url: str = ""
    image_caption: str = ""
    image_description: str = ""
    image_type: str = ""
    width: int = 0
    height: int = 0
    bbox: Optional[BBox] = None


class ProcessingError(BaseModel):
    message: str
    stage: str = ""
    details: dict = Field(default_factory=dict)


class ProcessingMetadata(BaseModel):
    ocr_applied: bool = False
    ocr_confidence: float = 0.0
    total_chunks: int = 0
    total_tables: int = 0
    total_images: int = 0
    processing_time_seconds: float = 0.0
    errors: List[ProcessingError] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ETLDocument(BaseModel):
    doc_id: str
    filename: str
    file_metadata: FileMetadata
    file_summary: FileSummary = Field(default_factory=FileSummary)
    pages: List[Page] = Field(default_factory=list)
    keyword_extraction: KeywordExtraction = Field(default_factory=KeywordExtraction)
    synthetic_qa_pairs: List[QAPair] = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)
    images: List[Image] = Field(default_factory=list)
    processing_metadata: ProcessingMetadata = Field(default_factory=ProcessingMetadata)
