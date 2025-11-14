import os
import json
import argparse
from typing import List, Dict, Any, Tuple

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError


def detect_lang_tags(text: str) -> List[str]:
    """
    Detect presence of English and Traditional Chinese scripts in raw text.
    - zh-Hant: if any CJK Unified Ideographs are present
    - en: if any Latin letters A–Z/a–z are present
    """
    import re
    tags: List[str] = []
    s = text or ""
    if re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]", s):
        tags.append("zh-Hant")
    if re.search(r"[A-Za-z]", s):
        tags.append("en")
    return tags


def derive_counts(text: str) -> Tuple[int, int]:
    s = (text or "").strip()
    if not s:
        return 0, 0
    words = len(s.split())
    chars = len(s)
    return words, chars


def map_chunk_to_doc(file_meta: Dict[str, Any], chunk: Dict[str, Any], chunk_index: int) -> Dict[str, Any]:
    # IDs
    document_id = file_meta.get("document_id") or file_meta.get("doc_id") or chunk.get("document_id")
    if not document_id:
        # Fallback to ETL doc_id if available
        document_id = file_meta.get("knowledge_id") or file_meta.get("knowledge_traceid") or "unknown"
    doc_id = f"{document_id}_{chunk_index}"

    # Text
    raw_text = chunk.get("chunk_content") or chunk.get("content") or ""
    # BM25 field population strategy: always duplicate to preserve context
    content_en, content_tc = raw_text, raw_text
    word_count, char_count = derive_counts(raw_text)

    # Titles
    title_name_en = file_meta.get("title_name_en") or file_meta.get("knowledge_name")
    title_name_tc = file_meta.get("title_name_tc") or file_meta.get("knowledge_name")

    # Other metadata
    filename = file_meta.get("file_name") or file_meta.get("sys_file_name") or ""
    branch_name = file_meta.get("branch_name") or file_meta.get("library_name_en") or ""

    # Page number: try common keys
    page_number = (
        chunk.get("chunk_page_number")
        or chunk.get("page_number")
        or chunk.get("page")
        or chunk.get("chunk_page")
        or chunk.get("page_index")
        or 0
    )

    # Entities normalization (strings or dicts with text/label)
    entities: List[str] = []
    # Prefer nested ETL path: chunk_metadata.chunk_entities; fall back to common aliases
    meta = chunk.get("chunk_metadata") or {}
    raw_entities = (
        chunk.get("entities")
        or chunk.get("chunk_entities")
        or meta.get("chunk_entities")
        or (file_meta.get("keyword_extraction") or {}).get("entities")
        or []
    )
    if isinstance(raw_entities, list):
        for e in raw_entities:
            if isinstance(e, str):
                entities.append(e)
            elif isinstance(e, dict):
                label = e.get("label") or e.get("type")
                text = e.get("text") or e.get("name")
                if text:
                    entities.append(text if not label else f"{text}:{label}")

    # Vector
    vector = chunk.get("chunk_content_vector")

    # Lang tags
    lang_tags = detect_lang_tags(raw_text)

    doc = {
        "id": doc_id,
        "content_en": content_en,
        "content_tc": content_tc,
        "content_chunk_vector": vector,
        "title_name_en": title_name_en,
        "title_name_tc": title_name_tc,
        "filename": filename,
        "branch_name": branch_name,
        "document_id": document_id,
        "entities": entities,
        "page_number": page_number,
        "word_count": word_count,
        "char_count": char_count,
        "lang_tags": lang_tags,
    }

    # Remove None fields to avoid index errors
    return {k: v for k, v in doc.items() if v is not None}


def load_etl_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_docs_from_etl(etl_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    file_meta = etl_json.get("file_index_metadata", {})
    chunks = etl_json.get("chunk_data", [])
    docs: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        docs.append(map_chunk_to_doc(file_meta, ch, i))
    return docs


def upload_docs(endpoint: str, api_key: str, index_name: str, docs: List[Dict[str, Any]], batch_size: int = 1000) -> None:
    client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        try:
            result = client.upload_documents(batch)
            # Optionally inspect result statuses
        except HttpResponseError as e:
            print(f"Upload failed at batch {i}: {e}")  # noqa: T201
            raise


def main():
    parser = argparse.ArgumentParser(description="Upload ETL JSON chunks to simple bilingual index")
    parser.add_argument("--etl-dir", type=str, required=True, help="Directory containing ETL JSON files")
    parser.add_argument("--index", type=str, default=os.getenv("SEARCH_INDEX_NAME") or os.getenv("AZURE_SEARCH_INDEX_NAME"), help="Target index name")
    parser.add_argument("--batch-size", type=int, default=1000, help="Upload batch size")
    args = parser.parse_args()

    endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT") or os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("SEARCH_SERVICE_KEY") or os.getenv("AZURE_SEARCH_ADMIN_KEY") or os.getenv("AZURE_SEARCH_API_KEY")
    if not endpoint or not api_key or not args.index:
        raise RuntimeError("Missing SEARCH_SERVICE_ENDPOINT/KEY and index name in environment or args")

    # Collect files
    etl_dir = args.etl_dir
    files = [
        os.path.join(etl_dir, fn)
        for fn in os.listdir(etl_dir)
        if fn.lower().endswith(".json")
    ]
    if not files:
        raise RuntimeError(f"No JSON files found in {etl_dir}")

    # Build and upload
    all_docs: List[Dict[str, Any]] = []
    for path in files:
        try:
            etl_json = load_etl_file(path)
            docs = build_docs_from_etl(etl_json)
            all_docs.extend(docs)
        except Exception as e:
            print(f"Skipping {path} due to error: {e}")  # noqa: T201
            continue

    upload_docs(endpoint, api_key, args.index, all_docs, batch_size=args.batch_size)
    print(f"Uploaded {len(all_docs)} docs to index {args.index}")  # noqa: T201


if __name__ == "__main__":
    main()