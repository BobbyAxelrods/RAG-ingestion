

import argparse
import os
import json
from typing import Any, Dict

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI
import pandas as pd

from src.search.query_vector import VectorSearcher
from src.search.hybrid_search import HybridSearcher
from src.search.hirarchical_hybrid_search import HierarchicalHybridSearcher
from src.search.prompt import build_rag_messages

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def run_excel_rag(
    excel_path: str,
    sheet_name: str,
    query_column: str,
    output_path: str,
    top_citations: int,
    vector_k: int,
    text_target: str,
) -> str:
    # Setup Azure Search and Azure OpenAI clients from env
    endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT") or os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("SEARCH_SERVICE_KEY") or os.getenv("AZURE_SEARCH_API_KEY")
    index_name = os.getenv("SEARCH_INDEX_NAME") or os.getenv("AZURE_SEARCH_INDEX_NAME")
    if not endpoint or not api_key or not index_name:
        raise RuntimeError("Missing SEARCH_SERVICE_ENDPOINT/KEY/INDEX_NAME in environment")

    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))

    aoai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    aoai_key = os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    aoai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    aoai_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-02-01"
    if not aoai_endpoint or not aoai_key or not aoai_deployment:
        raise RuntimeError("Missing Azure OpenAI env (endpoint/key/deployment)")
    openai_client = AzureOpenAI(api_key=aoai_key, api_version=aoai_api_version, azure_endpoint=aoai_endpoint)
    embedding_service = VectorSearcher.generate_embedding(openai_client, deployment_name=aoai_deployment)
    vector_searcher = VectorSearcher.vector_search(search_client, openai_client, deployment_name=aoai_deployment)

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if query_column not in df.columns:
        raise RuntimeError(f"Query column '{query_column}' not found in sheet '{sheet_name}'. Columns: {list(df.columns)}")

    # Prepare output columns
    df["hybrid_response"] = ""
    df["hybrid_citations"] = ""
    df["vector_response"] = ""
    df["vector_citations"] = ""
    df["final_response"] = ""

    for idx, row in df.iterrows():
        q = str(row[query_column] or "").strip()
        if not q:
            continue
        # For now, use vector-only retrieval with your VectorSearcher class
        results = vector_searcher.vector_search(q, top_k=top_citations, vector_field=[os.getenv("VECTOR_FIELDS") or "chunk_content_vector"])
        # Format results into JSON-friendly dicts
        docs: list[dict] = []
        for r in results:
            try:
                d = {
                    "file_name": r.get("file_name"),
                    "chunk_page_number": r.get("chunk_page_number"),
                    "chunk_function_summary": r.get("chunk_function_summary"),
                    "chunk_content": r.get("chunk_content"),
                    "score": r.get("@search.score"),
                }
            except Exception:
                # Fallback if SDK returns objects; try attribute access
                d = {
                    "file_name": getattr(r, "file_name", None),
                    "chunk_page_number": getattr(r, "chunk_page_number", None),
                    "chunk_function_summary": getattr(r, "chunk_function_summary", None),
                    "chunk_content": getattr(r, "chunk_content", None),
                    "score": getattr(r, "@search.score", None),
                }
            docs.append(d)
        df.at[idx, "vector_response"] = ""
        df.at[idx, "vector_citations"] = json.dumps(docs, ensure_ascii=False)

    # Write output Excel
    df.to_excel(output_path, sheet_name=sheet_name, index=False)
    return output_path


def run_single_query(
    query: str,
    *,
    top_citations: int = 6,
    vector_k: int = 150,
    text_target: str = "content",
    mode: str = "vector",
) -> Dict[str, Any]:
    import re

    def _detect_language(q: str) -> str:
        # Basic detection: use zh-TW for Han characters, else en-US
        return "zh-TW" if re.search(r"[\u4e00-\u9fff]", q) else "en-US"

    def _parse_hints(q: str) -> Dict[str, Any]:
        # Extract document code like PA000141 and page range like p. 1-5
        hints: Dict[str, Any] = {"document_id": None, "page_start": None, "page_end": None, "branch": None}
        code_match = re.search(r"\b([A-Z]{2,}[0-9]{3,})\b", q)
        if code_match:
            hints["document_id"] = code_match.group(1)
        page_match = re.search(r"p\.?\s*(\d+)\s*[-–—]\s*(\d+)", q, flags=re.IGNORECASE)
        if page_match:
            hints["page_start"] = int(page_match.group(1))
            hints["page_end"] = int(page_match.group(2))
        if re.search(r"澳門|Macau", q, flags=re.IGNORECASE):
            hints["branch"] = "MACAU"
        return hints

    def _clean_query_for_embedding(q: str) -> str:
        # Remove codes and page markers and symbols that add noise for embeddings
        q = re.sub(r"\b([A-Z]{2,}[0-9]{3,})\b", " ", q)  # remove document codes
        q = re.sub(r"p\.?\s*\d+\s*[-–—]\s*\d+", " ", q, flags=re.IGNORECASE)  # remove page ranges
        q = q.replace("<", " ").replace(">", " ")
        q = re.sub(r"\s+", " ", q).strip()
        return q
    # Setup Azure Search and Azure OpenAI clients from env
    endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT") or os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("SEARCH_SERVICE_KEY") or os.getenv("AZURE_SEARCH_API_KEY")
    index_name = os.getenv("SEARCH_INDEX_NAME") or os.getenv("AZURE_SEARCH_INDEX_NAME")
    if not endpoint or not api_key or not index_name:
        raise RuntimeError("Missing SEARCH_SERVICE_ENDPOINT/KEY/INDEX_NAME in environment")

    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))

    aoai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    aoai_key = os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    aoai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    aoai_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-02-01"
    if not aoai_endpoint or not aoai_key or not aoai_deployment:
        raise RuntimeError("Missing Azure OpenAI env (endpoint/key/deployment)")
    openai_client = AzureOpenAI(api_key=aoai_key, api_version=aoai_api_version, azure_endpoint=aoai_endpoint)

    # Parse vector fields (comma-separated supported)
    vf_env = os.getenv("VECTOR_FIELDS")
    vector_fields = [f.strip() for f in (vf_env.split(",") if vf_env else ["chunk_content_vector"]) if f.strip()]

    # Derive search options from query
    hints = _parse_hints(query)
    query_lang = _detect_language(query)
    search_fields = ["chunk_function_summary"] if text_target == "summary" else ["chunk_content"]
    odata_filters: list[str] = []
    if hints.get("document_id"):
        odata_filters.append(f"document_id eq '{hints['document_id']}'")
    if hints.get("page_start") is not None and hints.get("page_end") is not None:
        odata_filters.append(f"chunk_page_number ge {hints['page_start']} and chunk_page_number le {hints['page_end']}")
    if hints.get("branch"):
        odata_filters.append(f"branch_name eq '{hints['branch']}'")
    filter_expression = " and ".join(odata_filters) if odata_filters else None

    if mode == "hybrid":
        hybrid_searcher = HybridSearcher(search_client, openai_client, deployment_name=aoai_deployment)
        results = hybrid_searcher.hybrid_search(
            query_text=query,
            top_k=top_citations,
            vector_fields=vector_fields,
            k_nearest=vector_k,
            search_fields=search_fields,
            query_language=query_lang,
            filter_expression=filter_expression,
            select_fields=["file_name", "chunk_page_number", "chunk_function_summary", "chunk_content"],
        )
        # Fallback: if filters return no results, retry without filters
        if not results:
            results = hybrid_searcher.hybrid_search(
                query_text=query,
                top_k=top_citations,
                vector_fields=vector_fields,
                k_nearest=vector_k,
                search_fields=search_fields,
                query_language=query_lang,
                filter_expression=None,
                select_fields=["file_name", "chunk_page_number", "chunk_function_summary", "chunk_content"],
            )
    elif mode == "hierarchical":
        semantic_config = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG")
        hh_searcher = HierarchicalHybridSearcher(
            search_client,
            openai_client,
            deployment_name=aoai_deployment,
            id_field=os.getenv("ID_FIELD") or "doc_id",
            vector_fields=vector_fields,
            bm25_search_fields=search_fields,
            query_language=query_lang,
            select_fields=["file_name", "chunk_page_number", "chunk_function_summary", "chunk_content"],
            use_semantic_reranking=bool(semantic_config),
            semantic_configuration_name=semantic_config,
        )
        stage1_count = int(os.getenv("HH_STAGE1_CANDIDATES") or 500)
        stage2_count = int(os.getenv("HH_STAGE2_CANDIDATES") or 50)
        results = hh_searcher.search(
            query,
            top=top_citations,
            stage1_candidates=stage1_count,
            stage2_candidates=stage2_count,
            filter_expression=filter_expression,
        )
        if not results and filter_expression:
            # Fallback: rerun without filters to avoid empty citations
            results = hh_searcher.search(
                query,
                top=top_citations,
                stage1_candidates=stage1_count,
                stage2_candidates=stage2_count,
                filter_expression=None,
            )
    else:
        vector_searcher = VectorSearcher(search_client, openai_client, deployment_name=aoai_deployment)
        cleaned = _clean_query_for_embedding(query)
        results = vector_searcher.vector_search(cleaned, top_k=top_citations, vector_field=vector_fields)
    # Format results into JSON-friendly dicts
    docs: list[dict] = []
    for r in results:
        try:
            d = {
                "file_name": r.get("file_name"),
                "chunk_page_number": r.get("chunk_page_number"),
                "chunk_function_summary": r.get("chunk_function_summary"),
                "chunk_content": r.get("chunk_content"),
                "score": r.get("@search.score"),
            }
        except Exception:
            d = {
                "file_name": getattr(r, "file_name", None),
                "chunk_page_number": getattr(r, "chunk_page_number", None),
                "chunk_function_summary": getattr(r, "chunk_function_summary", None),
                "chunk_content": getattr(r, "chunk_content", None),
                "score": getattr(r, "@search.score", None),
            }
        docs.append(d)
    # Build chat messages from top citations
    top_docs = docs[:top_citations]

    # Configure chat model (prefer CHAT_* vars; fallback to AZURE_OPENAI_*)
    chat_endpoint = os.getenv("CHAT_MODEL_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
    chat_key = os.getenv("CHAT_MODEL_API_KEY") or os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    chat_model = os.getenv("CHAT_MODEL") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    chat_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-02-01"
    answer_text = ""
    if chat_endpoint and chat_key and chat_model:
        chat_client = AzureOpenAI(api_key=chat_key, api_version=chat_api_version, azure_endpoint=chat_endpoint)
        messages = build_rag_messages(query, top_docs)
        try:
            resp = chat_client.chat.completions.create(
                model=chat_model,
                messages=messages,
                temperature=0.2,
                max_tokens=600,
            )
            answer_text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            answer_text = f"[Answer generation failed: {str(e)}]"
    else:
        answer_text = "[Chat model not configured in environment]"

    # Shape result key based on mode for clarity
    key = "vector" if mode == "vector" else ("hybrid" if mode == "hybrid" else "hierarchical")
    return {key: {"citations": docs}, "final": {"answer": answer_text}}


def main():
    parser = argparse.ArgumentParser(description="Search runner using VectorSearcher/HybridSearcher/Hierarchical (single query or Excel batch)")
    parser.add_argument("excel", nargs="?", help="Path to input Excel file (xlsx)")
    parser.add_argument("--query", type=str, default=None, help="Single query to run (skips Excel mode)")
    parser.add_argument("--sheet", type=str, default="Sheet1", help="Sheet name containing queries (default: Sheet1)")
    parser.add_argument("--query-column", type=str, default="query", help="Column name containing queries (default: query)")
    parser.add_argument("--output", type=str, default=None, help="Path to output Excel (default: alongside input with suffix)")
    parser.add_argument("--top", type=int, default=6, help="Number of citations to use per mode (default: 6)")
    parser.add_argument("--k", type=int, default=150, help="Vector k-nearest neighbors (default: 150)")
    parser.add_argument("--text-target", type=str, choices=["content", "summary"], default="content", help="Which text field to target for hybrid (default: content)")
    parser.add_argument("--mode", type=str, choices=["vector", "hybrid", "hierarchical"], default="vector", help="Search mode: vector-only, hybrid (BM25+vector), or hierarchical (3-stage)")

    args = parser.parse_args()
    # Single-query mode
    if args.query:
        result = run_single_query(
            args.query,
            top_citations=args.top,
            vector_k=args.k,
            text_target=args.text_target,
            mode=args.mode,
        )
        # Ensure UTF-8 stdout
        try:
            import sys
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # Excel batch mode
    if not args.excel:
        raise SystemExit("Provide either --query for single-run or an Excel path for batch mode.")

    in_path = args.excel
    out_path = args.output or os.path.splitext(in_path)[0] + "_rag_output.xlsx"

    final_path = run_excel_rag(
        excel_path=in_path,
        sheet_name=args.sheet,
        query_column=args.query_column,
        output_path=out_path,
        top_citations=args.top,
        vector_k=args.k,
        text_target=args.text_target,
    )
    print(f"Saved RAG results to: {final_path}")


if __name__ == "__main__":
    raise SystemExit(main())

