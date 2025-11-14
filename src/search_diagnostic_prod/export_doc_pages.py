import os
import json
import argparse
from typing import Dict, List, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient


def _get_env(primary: str, *alternates: str, default=None, required=False):
    for key in (primary, *alternates):
        val = os.getenv(key)
        if val:
            return val
    if required and default is None:
        alt = ", ".join(alternates) if alternates else ""
        raise RuntimeError(f"Missing environment variable: {primary}{f' (alternatives: {alt})' if alt else ''}")
    return default


def export_docs(index_name: str, page_size: int = 1000) -> Dict[str, Any]:
    """Export filename + page_number + document_id from the index.

    Returns an object containing:
      - index: index name
      - total_count: total documents matched
      - grouped: list of {document_id, filenames, pages, count}
      - flat: list of {id, document_id, filename, page_number}
    """

    endpoint = _get_env("SEARCH_SERVICE_ENDPOINT", "AZURE_SEARCH_ENDPOINT", required=True)
    api_key = _get_env("SEARCH_SERVICE_KEY", "AZURE_SEARCH_API_KEY", required=True)
    client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))

    select_fields = ["id", "document_id", "filename", "page_number"]

    # First call to get total count
    initial = client.search(
        search_text="*",
        select=select_fields,
        include_total_count=True,
        top=page_size,
    )
    total = getattr(initial, "get_count", lambda: None)() or 0

    flat: List[Dict[str, Any]] = []

    # Iterate pages using skip in steps of page_size
    if total and total > page_size:
        for skip in range(0, total, page_size):
            results = client.search(
                search_text="*",
                select=select_fields,
                top=page_size,
                skip=skip,
            )
            for r in results:
                flat.append({
                    "id": r.get("id"),
                    "document_id": r.get("document_id"),
                    "filename": r.get("filename"),
                    "page_number": r.get("page_number"),
                })
    else:
        for r in initial:
            flat.append({
                "id": r.get("id"),
                "document_id": r.get("document_id"),
                "filename": r.get("filename"),
                "page_number": r.get("page_number"),
            })

    # Group by document_id
    grouped_map: Dict[str, Dict[str, Any]] = {}
    for row in flat:
        doc_id = row.get("document_id") or ""
        fn = row.get("filename")
        pg = row.get("page_number")
        if doc_id not in grouped_map:
            grouped_map[doc_id] = {
                "document_id": doc_id,
                "filenames": [],
                "pages": [],
                "count": 0,
            }
        g = grouped_map[doc_id]
        if fn and fn not in g["filenames"]:
            g["filenames"].append(fn)
        if pg is not None:
            g["pages"].append(pg)
        g["count"] += 1

    # Sort pages and ensure uniqueness
    for g in grouped_map.values():
        g["pages"] = sorted(list({p for p in g["pages"] if p is not None}))

    # Sort by document_id
    grouped = sorted(grouped_map.values(), key=lambda x: (x["document_id"] or ""))

    return {
        "index": index_name,
        "total_count": len(flat),
        "grouped": grouped,
        "flat": flat,
    }


def main():
    parser = argparse.ArgumentParser(description="Export filename/page_number/document_id from Azure AI Search index, sorted by ID")
    parser.add_argument("--index", "-i", required=True, help="Index name to export from")
    parser.add_argument("--out", "-o", required=False, default=None, help="Output JSON path (default: docs_pages_<index>.json)")
    parser.add_argument("--flat", action="store_true", help="Write only flat list (omit grouped)")
    parser.add_argument("--page-size", type=int, default=1000, help="Page size for listing documents")
    args = parser.parse_args()

    data = export_docs(args.index, page_size=args.page_size)

    if args.flat:
        payload = data["flat"]
    else:
        payload = data

    out_path = args.out or f"docs_pages_{args.index}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {('flat list' if args.flat else 'report')} to {out_path}")


if __name__ == "__main__":
    main()