import os
import json
import argparse
import requests
import sys


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def _generate_query_embedding(text: str) -> list | None:
    """Client-side embedding using Azure OpenAI embeddings.
    Required env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT_NAME.
    Optional: AZURE_OPENAI_API_VERSION (default 2024-02-01).
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or os.getenv("OPENAI_EMBEDDING_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-02-01"
    if not endpoint or not api_key or not deployment:
        return None
    url = endpoint.rstrip("/") + f"/openai/deployments/{deployment}/embeddings?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": api_key}
    body = {"input": text}
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(body))
        if resp.status_code != 200:
            return None
        payload = resp.json()
        emb = (payload.get("data") or [{}])[0].get("embedding")
        return emb if isinstance(emb, list) else None
    except Exception:
        return None


def hybrid_search(index_name: str,
                  search_text: str,
                  search_fields: str = "chunk_content",
                  vector_fields: str = "chunk_content_vector",
                  select: str | None = None,
                  top: int = 10,
                  k: int = 50,
                  filter_expr: str | None = None,
                  semantic_config: str | None = None,
                  api_version: str | None = None,
                  exhaustive: bool | None = None,
                  oversampling: int | None = None,
                  vector_filter_mode: str | None = None,
                  facets: list[str] | None = None,
                  mode: str = "hybrid"):
    """Execute a hybrid query using Azure AI Search.

    - Uses 'vectorQueries' for 2025 API, 'vectors' for 2024 API, and 'vector' for older.
    - Enables semantic ranking when a semantic configuration is provided.
    """
    endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT") or os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("SEARCH_SERVICE_KEY") or os.getenv("AZURE_SEARCH_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError("Missing SEARCH_SERVICE_ENDPOINT or SEARCH_SERVICE_KEY in environment")

    api_version = api_version or os.getenv("SEARCH_API_VERSION") or os.getenv("AZURE_SEARCH_API_VERSION") or "2024-07-01"
    url = endpoint.rstrip("/") + f"/indexes/{index_name}/docs/search?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": api_key}

    # For vector-only mode, suppress keyword influence by using empty search
    body: dict = {
        "count": True,
        "search": ("" if mode == "vector" else search_text),
        "top": top,
    }
    if select:
        body["select"] = select
    if filter_expr:
        body["filter"] = filter_expr
    if facets:
        body["facets"] = facets
    # Only include searchFields when keyword is involved
    if search_fields and mode in ("hybrid", "keyword"):
        body["searchFields"] = search_fields

    # Semantic ranking (optional)
    if semantic_config and mode != "vector":
        body["queryType"] = "semantic"
        # Omit queryLanguage for broad compatibility; semantic config typically encodes language
        body["semanticConfiguration"] = semantic_config

    # Client-side embedding for the query
    # Only perform vector retrieval in hybrid or vector mode
    emb = _generate_query_embedding(search_text) if mode in ("hybrid", "vector") else None
    if emb and mode in ("hybrid", "vector"):
        if api_version.startswith("2025"):
            # Newest shape: vectorQueries
            vq = {
                "kind": "vector",
                "vector": emb,
                "k": k,
                "fields": vector_fields,
            }
            if exhaustive is not None:
                vq["exhaustive"] = exhaustive
            # Only include oversampling when explicitly enabled via env (compression-supported indexes)
            allow_oversampling = (os.getenv("VECTOR_OVERSAMPLING_ENABLED", "").lower() in ("1", "true", "yes"))
            if oversampling is not None and allow_oversampling:
                vq["oversampling"] = oversampling
            body["vectorQueries"] = [vq]
            if vector_filter_mode:
                body["vectorFilterMode"] = vector_filter_mode
        elif api_version.startswith("2024"):
            # 2024 shape: vectors
            body["vectors"] = [{"value": emb, "fields": vector_fields, "k": k}]
        else:
            # Older shape: single vector
            body["vector"] = {"value": emb, "fields": vector_fields, "k": k}

    resp = requests.post(url, headers=headers, data=json.dumps(body))
    if resp.status_code != 200:
        print(f"Search failed: {resp.status_code}\n{resp.text}")
        return None
    return resp.json()


def main():
    load_env()
    parser = argparse.ArgumentParser(description="Search client for Azure AI Search (hybrid/keyword/vector)")
    parser.add_argument("index", help="Target Azure Search index name")
    parser.add_argument("search", help="Search text")
    parser.add_argument("--select", type=str, default=None, help="Comma-separated fields to return")
    parser.add_argument("--top", type=int, default=10, help="Number of results to return (default: 10)")
    parser.add_argument("--search-fields", type=str, default=None, help="Text fields to target; overrides --text-target")
    parser.add_argument("--vector-fields", type=str, default="chunk_content_vector", help="Vector field(s) to target (default: chunk_content_vector)")
    parser.add_argument("--k", type=int, default=50, help="TopK for vector retrieval (default: 50)")
    parser.add_argument("--filter", type=str, default=None, help="OData filter expression (optional)")
    parser.add_argument("--semantic-config", type=str, default=None, help="Semantic configuration name (optional)")
    parser.add_argument("--api-version", type=str, default=None, help="API version, e.g., 2024-07-01 or 2025-09-01")
    parser.add_argument("--exhaustive", action="store_true", help="Use exhaustive vector search (2025 API)")
    parser.add_argument("--oversampling", type=int, default=None, help="Oversampling for vector queries (2025 API)")
    parser.add_argument("--vector-filter-mode", type=str, default=None, help="Vector filter mode, e.g., postFilter (2025 API)")
    parser.add_argument("--facets", type=str, nargs="*", default=None, help="Facet fields (optional)")
    parser.add_argument("--output", type=str, default=None, help="Write result JSON to this path (UTF-8)")
    parser.add_argument("--mode", type=str, choices=["hybrid", "keyword", "vector"], default="hybrid", help="Query mode: hybrid, keyword-only, or vector-only")
    parser.add_argument("--text-target", type=str, choices=["content", "summary"], default="content", help="Target text field: chunk content or function summary")
    parser.add_argument("--sort-by", type=str, choices=["auto", "score", "reranker", "none"], default="auto", help="Client-side sort: auto (prefer reranker), score, reranker, or none")
    parser.add_argument("--sort-order", type=str, choices=["asc", "desc"], default="desc", help="Client-side sort order (default: desc)")

    args = parser.parse_args()
    # Determine effective text field target
    effective_search_fields = args.search_fields
    if not effective_search_fields:
        effective_search_fields = "chunk_function_summary" if args.text_target == "summary" else "chunk_content"

    result = hybrid_search(index_name=args.index,
                           search_text=args.search,
                           search_fields=effective_search_fields,
                           vector_fields=args.vector_fields,
                           select=args.select,
                           top=args.top,
                           k=args.k,
                           filter_expr=args.filter,
                           semantic_config=args.semantic_config,
                           api_version=args.api_version,
                           exhaustive=True if args.exhaustive else None,
                           oversampling=args.oversampling,
                           vector_filter_mode=args.vector_filter_mode,
                           facets=args.facets,
                           mode=args.mode)
    if not result:
        return 1
    # Client-side sorting for evaluation convenience
    try:
        if args.sort_by != "none" and isinstance(result.get("value"), list):
            reverse = (args.sort_order == "desc")
            def pick_score(r: dict):
                if args.sort_by == "score":
                    return r.get("@search.score", 0)
                if args.sort_by == "reranker":
                    return r.get("@search.rerankerScore", 0)
                # auto: prefer rerankerScore if present else score
                return r.get("@search.rerankerScore", r.get("@search.score", 0))
            result["value"].sort(key=pick_score, reverse=reverse)
    except Exception:
        pass
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to write output JSON: {e}")
            return 1
    else:
        # Ensure UTF-8 capable stdout where supported
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())