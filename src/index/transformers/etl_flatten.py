from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Iterable, List


def _iso_to_dt(value: str | None) -> str | None:
    if not value:
        return None
    try:
        # Normalize to ISO 8601 with timezone Z if missing
        # Return original if it already looks ISO-like
        return value
    except Exception:
        return None


def flatten_etl_json(etl_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert nested ETL JSON (document-level with chunk_data list)
    into flat documents matching the aisearch_index_schema.json fields.

    Produces one search document per chunk.
    """
    out: List[Dict[str, Any]] = []

    doc_id = etl_json.get("doc_id") or etl_json.get("document_id")
    sys_meta = etl_json.get("system_file_metadata") or {}
    file_meta = etl_json.get("file_index_metadata") or {}
    chunks: List[Dict[str, Any]] = etl_json.get("chunk_data") or []

    sys_file_name = sys_meta.get("file_name") or file_meta.get("file_name")
    sys_file_path = sys_meta.get("file_path") or file_meta.get("file_path")
    sys_file_size_bytes = sys_meta.get("file_size_bytes") or file_meta.get("file_size_bytes")
    sys_file_type = sys_meta.get("file_type") or file_meta.get("file_type")
    sys_last_updated = _iso_to_dt(sys_meta.get("last_updated") or file_meta.get("last_updated"))
    sys_page_count = sys_meta.get("page_count") or file_meta.get("page_count")
    sys_extracted_at = _iso_to_dt(etl_json.get("extracted_at") or sys_meta.get("extracted_at"))
    sys_processing_version = etl_json.get("processing_version") or sys_meta.get("processing_version")

    display_file_name = file_meta.get("display_file_name") or sys_file_name
    library_name_en = file_meta.get("library_name_en")
    library_name_tc = file_meta.get("library_name_tc")
    category_name_en = file_meta.get("category_name_en")
    category_name_tc = file_meta.get("category_name_tc")
    title_name_en = file_meta.get("title_name_en") or file_meta.get("title_en")
    title_name_tc = file_meta.get("title_name_tc") or file_meta.get("title_tc")
    file_url = file_meta.get("file_url")
    branch_name = file_meta.get("branch_name")
    item_type = file_meta.get("item_type")
    item_url = file_meta.get("item_url")

    for idx, chunk in enumerate(chunks):
        chunk_content = chunk.get("chunk_content")
        chunk_vector = chunk.get("chunk_content_vector")
        meta = chunk.get("chunk_metadata") or {}

        chunk_page_number = meta.get("chunk_page_number") or chunk.get("page_number") or (idx + 1)
        chunk_function_summary = meta.get("chunk_function_summary") or meta.get("function_summary")
        chunk_char_count = meta.get("chunk_char_count")
        chunk_word_count = meta.get("chunk_word_count")
        chunk_sentence_count = meta.get("chunk_sentence_count")
        chunk_entities = meta.get("chunk_entities") or []

        # QA pairs may be in meta["page_qna_pairs"] or under chunk["qna_pairs"]
        qna_pairs = meta.get("page_qna_pairs") or chunk.get("qna_pairs") or []
        qa_questions: List[str] = []
        qa_answers: List[str] = []
        qa_confidences: List[float] = []
        for qp in qna_pairs:
            q = qp.get("question")
            a = qp.get("answer")
            c = qp.get("confidence") or qp.get("score")
            if q:
                qa_questions.append(q)
            if a:
                qa_answers.append(a)
            if isinstance(c, (int, float)):
                qa_confidences.append(float(c))

        qa_confidence = sum(qa_confidences) / len(qa_confidences) if qa_confidences else 0.0

        flat_doc: Dict[str, Any] = {
            "doc_id": f"{doc_id}-{chunk_page_number}",
            "sys_file_name": sys_file_name,
            "sys_file_path": sys_file_path,
            "sys_file_size_bytes": sys_file_size_bytes,
            "sys_file_type": sys_file_type,
            "sys_last_updated": sys_last_updated,
            "sys_page_count": sys_page_count,
            "sys_extracted_at": sys_extracted_at,
            "sys_processing_version": sys_processing_version,
            "file_name": display_file_name,
            "library_name_en": library_name_en,
            "library_name_tc": library_name_tc,
            "category_name_en": category_name_en,
            "category_name_tc": category_name_tc,
            "title_name_en": title_name_en,
            "title_name_tc": title_name_tc,
            "file_url": file_url,
            "branch_name": branch_name,
            "item_type": item_type,
            "item_url": item_url,
            "chunk_content": chunk_content,
            "chunk_content_vector": chunk_vector,
            "chunk_page_number": chunk_page_number,
            "chunk_function_summary": chunk_function_summary,
            "chunk_char_count": chunk_char_count,
            "chunk_word_count": chunk_word_count,
            "chunk_sentence_count": chunk_sentence_count,
            "chunk_entities": chunk_entities,
            "qa_questions": qa_questions,
            "qa_answers": qa_answers,
            "qa_confidence": qa_confidence,
        }

        out.append(flat_doc)

    return out