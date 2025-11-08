"""
Extraction schema writer service.

Produces JSON aligned with `project_plan/etl_schema.json` and
`etl_schema_sample.json`, focusing on chunk-level extraction payloads.

Top-level fields:
- doc_id
- system_file_metadata (sys_*)
- file_index_metadata
- chunk_data[] (array of chunk entries)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Dict

from pydantic import BaseModel

from src.etl.models.document_models import SearchDocument, parse_chunk_id
from src.etl.models.etl_models import Chunk as ETLChunk
from src.etl.services.page_insights_service import PageInsightsService


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if isinstance(dt, datetime) else None


def _tokenize(text: str) -> tuple[int, int]:
    words = [w for w in (text or "").split() if w.strip()]
    sentences = [s for s in (text or "").split(".") if s.strip()]
    return len(words), len(sentences)


def _deterministic_doc_id(source: str) -> str:
    import hashlib

    h = hashlib.sha1((source or "").encode("utf-8")).hexdigest()[:12]
    return f"doc_{h}"


@dataclass
class QAPair:
    question: str
    answer: str
    qa_doc_id: str
    qa_page_number: int
    qa_chunk_ids: List[str]
    qa_confidence: float


@dataclass
class ChunkMetadata:
    chunk_page_number: int
    chunk_function_summary: str
    chunk_char_count: int
    chunk_word_count: int
    chunk_sentence_count: int
    chunk_entities: List[str]
    page_qna_pairs: List[QAPair]


@dataclass
class ChunkData:
    chunk_content: str
    chunk_content_vector: List[float]
    chunk_metadata: ChunkMetadata


@dataclass
class SystemFileMetadata:
    sys_file_name: str
    sys_file_path: str
    sys_file_size_bytes: int
    sys_file_type: str
    sys_last_updated: Optional[str]
    sys_page_count: int
    sys_extracted_at: str
    sys_processing_version: str


@dataclass
class FileIndexMetadata:
    file_name: str
    library_name_en: str
    library_name_tc: str
    category_name_en: str
    category_name_tc: str
    title_name_en: str
    title_name_tc: str
    file_url: str
    branch_name: str
    item_type: str
    item_url: str


class ExtractionDocument(BaseModel):
    doc_id: str
    system_file_metadata: dict
    file_index_metadata: dict
    chunk_data: List[dict]


class ExtractionWriter(BaseModel):
    processing_version: str = "v1.0"

    def build_extraction_document(
        self,
        filename: str,
        search_documents: List[SearchDocument],
        original_path: Optional[str] = None,
    ) -> ExtractionDocument:
        # System metadata
        source_path = original_path or filename
        sys_file_name = Path(filename).name
        sys_file_type = Path(filename).suffix.lower().lstrip(".")
        sys_file_path = source_path
        sys_file_size = 0
        sys_last_updated: Optional[str] = None

        try:
            if original_path and os.path.exists(original_path):
                sys_file_size = os.path.getsize(original_path)
                sys_last_updated = _iso(datetime.fromtimestamp(os.path.getmtime(original_path)))
        except Exception:
            pass

        # Read page_count and build file_index_metadata from metadata_file of first doc
        sys_page_count = 0
        file_index: Dict[str, Any] = {}
        if search_documents:
            try:
                mf = json.loads(search_documents[0].metadata_file)
                # Use total pages for system metadata
                sys_page_count = int(mf.get("total_pages", 0) or 0)

                # Start with all fields from metadata_file to include all CSV/Excel columns
                # This ensures the output JSON contains the complete metadata row
                file_index.update(mf)

                # Normalize common keys expected by downstream consumers, without overwriting existing data
                # file_name: prefer explicit file_name, fall back to filename or Path stem
                file_index.setdefault("file_name", mf.get("file_name") or mf.get("filename") or Path(filename).stem)

                # title_name: map possible aliases (title_en/title_tc) to title_name_en/title_name_tc
                file_index.setdefault("title_name_en", mf.get("title_name_en") or mf.get("title_en") or Path(filename).stem)
                file_index.setdefault("title_name_tc", mf.get("title_name_tc") or mf.get("title_tc") or "")

                # Library/category bilingual names
                file_index.setdefault("library_name_en", mf.get("library_name_en") or "")
                file_index.setdefault("library_name_tc", mf.get("library_name_tc") or "")
                file_index.setdefault("category_name_en", mf.get("category_name_en") or "")
                file_index.setdefault("category_name_tc", mf.get("category_name_tc") or "")

                # URL/type/branch when available; leave empty if not present
                file_index.setdefault("file_url", mf.get("file_url") or "")
                file_index.setdefault("branch_name", mf.get("branch_name") or "")
                file_index.setdefault("item_type", mf.get("item_type") or "")
                file_index.setdefault("item_url", mf.get("item_url") or "")

                # Deduplicate alias keys: if canonical CSV-style keys exist, drop derived aliases
                # Prefer: file_name over filename; title_name_en over title_en; title_name_tc over title_tc
                if "file_name" in file_index and "filename" in file_index:
                    try:
                        del file_index["filename"]
                    except Exception:
                        pass
                if "title_name_en" in file_index and "title_en" in file_index:
                    try:
                        del file_index["title_en"]
                    except Exception:
                        pass
                if "title_name_tc" in file_index and "title_tc" in file_index:
                    try:
                        del file_index["title_tc"]
                    except Exception:
                        pass
            except Exception:
                # If metadata_file is not present or invalid, populate minimal defaults
                file_index = {
                    "file_name": Path(filename).stem,
                    "library_name_en": "",
                    "library_name_tc": "",
                    "category_name_en": "",
                    "category_name_tc": "",
                    "title_name_en": Path(filename).stem,
                    "title_name_tc": "",
                    "file_url": "",
                    "branch_name": "",
                    "item_type": "",
                    "item_url": "",
                }

        sys_meta = SystemFileMetadata(
            sys_file_name=sys_file_name,
            sys_file_path=sys_file_path,
            sys_file_size_bytes=sys_file_size,
            sys_file_type=sys_file_type,
            sys_last_updated=sys_last_updated,
            sys_page_count=sys_page_count,
            sys_extracted_at=_iso(datetime.utcnow()) or "",
            sys_processing_version=self.processing_version,
        )

        # Build chunk_data array
        # Group by page_number via chunk_id parsing
        chunk_entries: List[dict] = []
        base_doc_id = _deterministic_doc_id(sys_file_path)

        # Optional: simple page-level Q&A derived from summary (kept for fallback)
        file_summary_text = search_documents[0].file_summary if search_documents else ""

        # Pre-compute per-page LLM insights (entities + Q&A)
        # Map: page_number -> {
        #   "entities": List[str],
        #   "qas": List[QAPair]
        # }
        pages_map: Dict[int, List[SearchDocument]] = {}
        for sd in search_documents:
            _, page_number, _ = parse_chunk_id(sd.id)
            page_num = int(page_number or 0)
            pages_map.setdefault(page_num, []).append(sd)

        insights_service = PageInsightsService()

        def _clean_terms(items: List[str]) -> List[str]:
            cleaned: List[str] = []
            for it in items:
                if not it:
                    continue
                t = it.strip()
                # Remove trailing punctuation that indicates truncated phrases
                t = t.rstrip("-•,:；，：…")
                # Basic length filter to avoid fragments
                if len(t) < 2:
                    continue
                cleaned.append(t)
            # Deduplicate preserving order
            seen = set()
            uniq: List[str] = []
            for t in cleaned:
                if t not in seen:
                    seen.add(t)
                    uniq.append(t)
            return uniq

        page_results: Dict[int, Dict[str, Any]] = {}
        for page_num, sds in pages_map.items():
            # Aggregate page text
            page_text = "\n".join([sd.content_chunk or "" for sd in sds])
            # Build ETL chunks for Q&A generation
            etl_chunks: List[ETLChunk] = []
            for sd in sds:
                _, _, chunk_index = parse_chunk_id(sd.id)
                etl_chunks.append(
                    ETLChunk(
                        chunk_id=sd.id,
                        chunk_text=sd.content_chunk or "",
                        chunk_position=int(chunk_index),
                        content_chunk_vector=sd.content_chunk_dim,
                    )
                )
            try:
                result = insights_service.analyze_page(
                    page_text=page_text,
                    filename=sys_file_name,
                    doc_id=f"{base_doc_id}_p{page_num:04d}",
                    page_number=page_num,
                    chunks=etl_chunks,
                )
                # Union of categorized entities for chunk_entities
                cats = result.categories or {}
                union_entities = _clean_terms(
                    (cats.get("insurance_product", [])
                     + cats.get("insurance_term", [])
                     + cats.get("location", [])
                     + cats.get("people_occupation_user", []))
                )
                # Map QAPair from etl_models to writer dataclass
                qas: List[QAPair] = []
                for qa in result.qna_pairs:
                    qas.append(
                        QAPair(
                            question=qa.question,
                            answer=qa.answer,
                            qa_doc_id=qa.qa_doc_id,
                            qa_page_number=qa.qa_page_number,
                            qa_chunk_ids=qa.qa_chunk_ids,
                            qa_confidence=qa.qa_confidence,
                        )
                    )
                page_results[page_num] = {"entities": union_entities, "qas": qas}
            except Exception:
                # Fallback: no entities/QAs
                page_results[page_num] = {"entities": [], "qas": []}

        for sd in search_documents:
            _, page_number, _ = parse_chunk_id(sd.id)
            page_num = int(page_number or 0)
            char_count = len(sd.content_chunk or "")
            word_count, sentence_count = _tokenize(sd.content_chunk or "")

            # Use LLM-generated Q&A and entities for this page; do not insert generic summary-based Q&A
            page_insights = page_results.get(page_num, {"entities": [], "qas": []})
            qas: List[QAPair] = page_insights.get("qas", [])
            # Per requirements, only include Q&A pairs that are specific and answerable from the text.
            # If none are generated for this page, leave the array empty.

            cmd = ChunkMetadata(
                chunk_page_number=page_num,
                chunk_function_summary=file_summary_text,
                chunk_char_count=char_count,
                chunk_word_count=word_count,
                chunk_sentence_count=sentence_count,
                chunk_entities=page_insights.get("entities", []),
                page_qna_pairs=qas,
            )

            cdata = ChunkData(
                chunk_content=sd.content_chunk,
                chunk_content_vector=sd.content_chunk_dim,
                chunk_metadata=cmd,
            )

            chunk_entries.append(
                {
                    "chunk_content": cdata.chunk_content,
                    "chunk_content_vector": cdata.chunk_content_vector,
                    "chunk_metadata": {
                        "chunk_page_number": cmd.chunk_page_number,
                        "chunk_function_summary": cmd.chunk_function_summary,
                        "chunk_char_count": cmd.chunk_char_count,
                        "chunk_word_count": cmd.chunk_word_count,
                        "chunk_sentence_count": cmd.chunk_sentence_count,
                        "chunk_entities": cmd.chunk_entities,
                        "page_qna_pairs": [asdict(q) for q in cmd.page_qna_pairs],
                    },
                }
            )

        return ExtractionDocument(
            doc_id=f"{base_doc_id}",
            system_file_metadata=asdict(sys_meta),
            file_index_metadata=file_index,
            chunk_data=chunk_entries,
        )

    def write_extraction_json(
        self,
        filename: str,
        search_documents: List[SearchDocument],
        output_path: Path,
        original_path: Optional[str] = None,
    ) -> Path:
        doc = self.build_extraction_document(filename, search_documents, original_path=original_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = doc.model_dump()
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return output_path