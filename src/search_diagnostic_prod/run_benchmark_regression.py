import os
import sys
import json
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from src.search_prod.vector_search import AzureVectorSearch
except Exception as e:
    raise RuntimeError(f"Failed to import AzureVectorSearch: {e}")

try:
    from src.search_prod.hybrid_search import AzureHybridSearch
except Exception as e:
    raise RuntimeError(f"Failed to import AzureHybridSearch: {e}")

try:
    from src.search_prod.simple_search import AzureSimpleSearch
except Exception as e:
    raise RuntimeError(f"Failed to import AzureSimpleSearch: {e}")


def _standardize_simple_payload(query: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    normalized: List[Dict[str, Any]] = []
    for r in items:
        normalized.append({
            "id": r.get("document_id") or r.get("doc_id") or r.get("id"),
            "document_id": r.get("document_id") or r.get("doc_id") or r.get("id"),
            "title_name_en": r.get("title_name_en") or r.get("title") or "",
            "title_name_tc": r.get("title_name_tc") or "",
            "content_en": r.get("content_en") or "",
            "content_tc": r.get("content_tc") or "",
            "filename": r.get("file_name") or r.get("filename"),
            "page_number": r.get("page") or r.get("chunk_page_number") or r.get("page_number"),
            "score": r.get("score") or r.get("@search.score"),
            "content_chunk": r.get("chunk_content") or r.get("content") or "",
        })
    return {
        "query": query,
        "search_method": "simple",
        "total_result": len(normalized),
        "results": normalized,
    }


def _build_clients(index_name: Optional[str] = None) -> Tuple[SearchClient, AzureOpenAI]:
    endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT") or os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("SEARCH_SERVICE_KEY") or os.getenv("AZURE_SEARCH_API_KEY")
    idx = index_name or os.getenv("SEARCH_INDEX_NAME") or os.getenv("AZURE_SEARCH_INDEX_NAME")
    if not endpoint or not api_key or not idx:
        raise RuntimeError("Missing SEARCH_SERVICE_ENDPOINT/KEY/INDEX_NAME in environment")

    search_client = SearchClient(endpoint=endpoint, index_name=idx, credential=AzureKeyCredential(api_key))

    aoai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    aoai_key = os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    aoai_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-02-01"
    if not aoai_endpoint or not aoai_key:
        raise RuntimeError("Missing Azure OpenAI env (endpoint/key)")
    openai_client = AzureOpenAI(api_key=aoai_key, api_version=aoai_api_version, azure_endpoint=aoai_endpoint)
    return search_client, openai_client


def _probe_index_fields(search_client: SearchClient, sample_size: int = 1) -> Dict[str, Any]:
    results = search_client.search(search_text="*", top=sample_size)
    sample = []
    for r in results:
        sample.append(dict(r))
    keys = list(sample[0].keys()) if sample else []
    return {"keys": keys, "sample": sample}


def _detect_columns(df: pd.DataFrame) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            key = cand.lower().strip()
            if key in cols:
                return cols[key]
        # try fuzzy contains
        for k, orig in cols.items():
            for cand in candidates:
                if cand.lower() in k:
                    return orig
        return None

    qid_col = pick(["Question ID", "question_id", "qid", "id"]) or list(df.columns)[0]
    en_col = pick(["English", "Question English", "EN", "query_en", "english query", "en_question"]) or None
    tc_col = pick(["Traditional Chinese", "Chinese", "TC", "query_tc", "tc_question", "zh-hant"]) or None
    ref_docs_col = pick(["reference_documents", "Reference Documents", "references", "ref_docs"]) or None
    ref_marker_col = pick(["reference_marker", "Reference Marker", "ref_marker", "ref_point"]) or None
    return qid_col, en_col, tc_col, ref_docs_col, ref_marker_col


def run_regression(excel_path: str,
                   sheet_name: Optional[str] = None,
                   index_name: Optional[str] = None,
                   top_k: int = 10,
                   output_root: str = "results_json",
                   probe: bool = False) -> Dict[str, Any]:
    search_client, openai_client = _build_clients(index_name)

    if probe:
        info = _probe_index_fields(search_client, sample_size=1)
        return {"probe": info}

    # Resolve effective index name, prefer override
    effective_index = index_name or os.getenv("SEARCH_INDEX_NAME") or os.getenv("AZURE_SEARCH_INDEX_NAME")
    vec = AzureVectorSearch(
        search_endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT") or os.getenv("AZURE_SEARCH_ENDPOINT"),
        search_key=os.getenv("SEARCH_SERVICE_KEY") or os.getenv("AZURE_SEARCH_API_KEY"),
        openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_key=os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY"),
        index_name=effective_index or "",
    )
    hyb = AzureHybridSearch(search_client, openai_client)
    bm25 = AzureSimpleSearch(search_client)

    df = pd.read_excel(excel_path, sheet_name=sheet_name) if sheet_name else pd.read_excel(excel_path)
    qid_col, en_col, tc_col, ref_docs_col, ref_marker_col = _detect_columns(df)

    os.makedirs(output_root, exist_ok=True)
    saved: List[str] = []

    processed = 0
    for _, row in df.iterrows():
        qid = str(row.get(qid_col) or "").strip()
        if not qid:
            continue
        ref_docs = row.get(ref_docs_col) if ref_docs_col else None
        ref_point = row.get(ref_marker_col) if ref_marker_col else None

        q_folder = os.path.join(output_root, qid)
        os.makedirs(q_folder, exist_ok=True)

        # English
        if en_col:
            en_query = row.get(en_col)
            if pd.notna(en_query) and str(en_query).strip():
                en_query = str(en_query).strip()
                en_vector = vec.vector_search(en_query, top_k=top_k, query_lang="en")
                en_hybrid = hyb.hybrid_search(en_query, top_k=top_k, query_lang="en")
                en_simple_items = bm25.simple_search(en_query, top_k=top_k, query_lang="en")
                en_simple = _standardize_simple_payload(en_query, en_simple_items)

                en_combined = {
                    "question_id": qid,
                    "language": "en",
                    "reference_documents": ref_docs,
                    "reference_marker": ref_point,
                    "query": en_query,
                    "runs": [en_vector, en_hybrid, en_simple],
                }
                out_path = os.path.join(q_folder, f"{qid}_en.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(en_combined, f, ensure_ascii=False, indent=2)
                saved.append(out_path)

        # Traditional Chinese
        if tc_col:
            tc_query = row.get(tc_col)
            if pd.notna(tc_query) and str(tc_query).strip():
                tc_query = str(tc_query).strip()
                tc_vector = vec.vector_search(tc_query, top_k=top_k, query_lang="tc")
                tc_hybrid = hyb.hybrid_search(tc_query, top_k=top_k, query_lang="tc")
                tc_simple_items = bm25.simple_search(tc_query, top_k=top_k, query_lang="tc")
                tc_simple = _standardize_simple_payload(tc_query, tc_simple_items)

                tc_combined = {
                    "question_id": qid,
                    "language": "tc",
                    "reference_documents": ref_docs,
                    "reference_marker": ref_point,
                    "query": tc_query,
                    "runs": [tc_vector, tc_hybrid, tc_simple],
                }
                out_path = os.path.join(q_folder, f"{qid}_tc.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(tc_combined, f, ensure_ascii=False, indent=2)
                saved.append(out_path)

        processed += 1
        limit_env = os.getenv("REGRESSION_LIMIT")
        limit = int(limit_env) if limit_env and limit_env.isdigit() else None
        if limit and processed >= limit:
            break

    return {"saved": saved, "output_root": output_root}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run benchmark regression to generate per-question JSONs (EN/TC)")
    parser.add_argument("--excel", default=os.path.join("artifact", "data_library", "evaluation", "benchmark_ii ( steven)", "reviewed_pru.xlsx"), help="Path to reviewed_pru.xlsx")
    parser.add_argument("--sheet", default=None, help="Sheet name in Excel")
    parser.add_argument("--index", default=os.getenv("AZURE_SEARCH_INDEX_NAME") or os.getenv("SEARCH_INDEX_NAME") or None, help="Index name override")
    parser.add_argument("--top", type=int, default=10, help="Top K per search method")
    parser.add_argument("--out", default="results_json", help="Root folder to write per-question JSONs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to process")
    parser.add_argument("--probe", action="store_true", help="Probe index to list available fields and a sample record")
    args = parser.parse_args()

    if args.limit is not None:
        os.environ["REGRESSION_LIMIT"] = str(args.limit)
    result = run_regression(args.excel, sheet_name=args.sheet, index_name=args.index, top_k=args.top, output_root=args.out, probe=args.probe)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()