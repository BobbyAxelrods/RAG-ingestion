import os
import json
import argparse
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient


def _get_env(*names: str, required: bool = False, default: Optional[str] = None) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    if required and default is None:
        raise RuntimeError(f"Missing required environment variable: one of {', '.join(names)}")
    return default


def build_clients(index_name: str) -> tuple[SearchIndexClient, SearchClient]:
    endpoint = _get_env("SEARCH_SERVICE_ENDPOINT", "AZURE_SEARCH_ENDPOINT", required=True)
    api_key = _get_env("SEARCH_SERVICE_KEY", "AZURE_SEARCH_API_KEY", required=True)
    index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))
    return index_client, search_client


def serialize_field(f: Any) -> Dict[str, Any]:
    # Safe serialization of SearchField properties across SDK versions
    return {
        "name": getattr(f, "name", None),
        "type": getattr(f, "type", None),
        "searchable": getattr(f, "searchable", None),
        "filterable": getattr(f, "filterable", None),
        "sortable": getattr(f, "sortable", None),
        "facetable": getattr(f, "facetable", None),
        "retrievable": getattr(f, "retrievable", None),
        # Analyzer properties vary across SDK versions; capture all common variants
        "analyzer": getattr(f, "analyzer", None),
        "index_analyzer": getattr(f, "index_analyzer", None),
        "search_analyzer": getattr(f, "search_analyzer", None),
        "analyzer_name": getattr(f, "analyzer_name", None),
        "index_analyzer_name": getattr(f, "index_analyzer_name", None),
        "search_analyzer_name": getattr(f, "search_analyzer_name", None),
        "synonym_map_names": getattr(f, "synonym_map_names", None),
        "vector_search_dimensions": getattr(f, "vector_search_dimensions", None),
        "vector_search_profile_name": getattr(f, "vector_search_profile_name", None),
    }


def serialize_vector_search(vs: Any) -> Dict[str, Any]:
    if not vs:
        return {}
    algos = []
    for a in getattr(vs, "algorithm_configurations", []) or []:
        algos.append({
            "name": getattr(a, "name", None),
            "kind": getattr(a, "kind", None),
            "parameters": {
                # HNSW parameters vary by SDK; capture common ones if present
                "m": getattr(getattr(a, "parameters", None), "m", None),
                "ef_construction": getattr(getattr(a, "parameters", None), "ef_construction", None),
                "metric": getattr(getattr(a, "parameters", None), "metric", None),
            },
        })
    profiles = []
    for p in getattr(vs, "profiles", []) or []:
        profiles.append({
            "name": getattr(p, "name", None),
            "algorithm_configuration_name": getattr(p, "algorithm_configuration_name", None),
            "vectorizer": getattr(p, "vectorizer", None),
        })
    return {"algorithm_configurations": algos, "profiles": profiles}


def serialize_semantic_settings(ss: Any) -> Dict[str, Any]:
    if not ss:
        return {}
    configs = []
    for c in getattr(ss, "configurations", []) or []:
        pf = getattr(c, "prioritized_fields", None)
        configs.append({
            "name": getattr(c, "name", None),
            "prioritized_fields": {
                "title_field": getattr(getattr(pf, "title_field", None), "field_name", None) if pf else None,
                "content_fields": [getattr(cf, "field_name", None) for cf in (getattr(pf, "content_fields", []) or [])] if pf else [],
                "keyword_fields": [getattr(kf, "field_name", None) for kf in (getattr(pf, "keyword_fields", []) or [])] if pf else [],
            },
        })
    return {
        "default_configuration": getattr(ss, "default_configuration", None),
        "configurations": configs,
    }


def pick_search_fields(index_fields: List[Dict[str, Any]]) -> List[str]:
    names = {f["name"] for f in index_fields}
    preferred = [
        "chunk_content",
        "content_en",
        "content_tc",
        "content",
        "title_en",
        "title_tc",
    ]
    chosen = [n for n in preferred if n in names]
    if chosen:
        return chosen
    # Fallback: any searchable string fields
    return [f["name"] for f in index_fields if f.get("type") == "Edm.String" and f.get("searchable")][:2]


def pick_select_fields(index_fields: List[Dict[str, Any]]) -> List[str]:
    names = {f["name"] for f in index_fields}
    preferred = [
        "id",
        "document_id",
        "filename",
        "file_name",
        "title_en",
        "title_tc",
        "content_en",
        "content_tc",
        "chunk_content",
        "page_number",
        "chunk_page_number",
        "branch_name",
        "entities",
    ]
    return [n for n in preferred if n in names]


def run_sample_queries(search_client: SearchClient, index_fields: List[Dict[str, Any]], queries: List[str], top: int = 5, filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    search_fields = pick_search_fields(index_fields)
    select_fields = pick_select_fields(index_fields)

    for q in queries:
        results_list: List[Dict[str, Any]] = []
        results = search_client.search(
            search_text=q,
            top=top,
            query_type="simple",
            search_fields=search_fields or None,
            select=select_fields or None,
            filter=filter_expr or None,
        )
        for r in results:
            entry: Dict[str, Any] = {}
            for f in select_fields:
                entry[f] = r.get(f)
            entry["@search.score"] = r.get("@search.score")
            # Always include a sensible id fallback
            entry["id"] = entry.get("id") or entry.get("document_id") or r.get("id")
            results_list.append(entry)
        samples.append({
            "query": q,
            "search_fields": search_fields,
            "select_fields": select_fields,
            "filter": filter_expr,
            "results": results_list,
        })
    return samples


def inspect_index(index_name: str, queries: Optional[List[str]] = None, top: int = 5, filter_expr: Optional[str] = None) -> Dict[str, Any]:
    index_client, search_client = build_clients(index_name)
    idx = index_client.get_index(index_name)

    fields = [serialize_field(f) for f in getattr(idx, "fields", [])]
    vector_search = serialize_vector_search(getattr(idx, "vector_search", None))
    semantic_settings = serialize_semantic_settings(getattr(idx, "semantic_settings", None))

    # Optional extras
    scoring_profiles = []
    for sp in getattr(idx, "scoring_profiles", []) or []:
        scoring_profiles.append({
            "name": getattr(sp, "name", None),
            "text": getattr(sp, "text", None),
            "functions": getattr(sp, "functions", None),
        })

    queries = queries or ["", "Payment", "保費"]
    samples = run_sample_queries(search_client, fields, queries, top=top, filter_expr=filter_expr)

    return {
        "index_name": index_name,
        "fields": fields,
        "vector_search": vector_search,
        "semantic_settings": semantic_settings,
        "scoring_profiles": scoring_profiles,
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect Azure AI Search index schema and sample data")
    parser.add_argument("--index", required=True, help="Target Azure Search index name")
    parser.add_argument("--out", required=False, default=None, help="Output report JSON path")
    parser.add_argument("--queries", required=False, default=None, help="Comma-separated sample queries (default: '',Payment,保費)")
    parser.add_argument("--top", type=int, default=5, help="Number of results per sample query")
    parser.add_argument("--filter", required=False, default=None, help="OData filter expression (e.g., document_id eq 'PA000141')")
    args = parser.parse_args()

    queries = [q.strip() for q in args.queries.split(",") if q.strip()] if args.queries else None
    report = inspect_index(args.index, queries=queries, top=args.top, filter_expr=args.filter)
    out_path = args.out or f"index_report_{args.index}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Index inspection report written to {out_path}")


if __name__ == "__main__":
    main()