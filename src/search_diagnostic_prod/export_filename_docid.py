import os
import csv
import argparse
from collections import Counter
from typing import Optional, List, Dict

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


def export_filename_docid(index_name: str, out_csv: str, summary_csv: Optional[str] = None, unique_out_csv: Optional[str] = None, batch_top: int = 1000) -> Dict[str, int]:
    endpoint = _get_env("SEARCH_SERVICE_ENDPOINT", "AZURE_SEARCH_ENDPOINT", required=True)
    api_key = _get_env("SEARCH_SERVICE_KEY", "AZURE_SEARCH_API_KEY", required=True)

    client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))
    index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    if summary_csv:
        os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)

    # Determine available select fields from index schema to avoid invalid $select errors
    idx = index_client.get_index(index_name)
    field_names = {getattr(f, "name", None) for f in getattr(idx, "fields", [])}
    select_fields: List[str] = ["document_id"]
    if "filename" in field_names:
        select_fields.append("filename")
    if "file_name" in field_names:
        select_fields.append("file_name")

    rows: List[Dict[str, str]] = []
    filename_counts: Counter[str] = Counter()

    # Using empty search text to enumerate all docs; the SDK paginates automatically
    pager = client.search(
        search_text="*",
        select=select_fields,
        include_total_count=True,
        query_type="simple",
    )

    for page in pager.by_page():
        for r in page:
            filename = (r.get("filename") or r.get("file_name") or "").strip()
            document_id = (r.get("document_id") or "").strip()
            rows.append({"filename": filename, "document_id": document_id})
            if filename:
                filename_counts[filename] += 1

    # Write detailed CSV (UTF-8 with BOM for Excel compatibility with TC/EN)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "document_id"])
        w.writeheader()
        w.writerows(rows)

    # Write summary counts CSV (filename, count) in UTF-8 with BOM
    if summary_csv:
        with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["filename", "count"])
            for fn, cnt in sorted(filename_counts.items(), key=lambda x: (-x[1], x[0])):
                w.writerow([fn, cnt])

    # Write unique filenames CSV (one filename per row) in UTF-8 with BOM
    if unique_out_csv:
        unique_fns = sorted(filename_counts.keys())
        with open(unique_out_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["filename"])  # single-column header
            for fn in unique_fns:
                w.writerow([fn])

    # Return quick stats
    return {
        "total_rows": len(rows),
        "unique_filenames": len(filename_counts),
        "max_chunks_per_file": max(filename_counts.values()) if filename_counts else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Export filename and document_id from Azure AI Search index to CSV, with per-file counts")
    parser.add_argument("--index", required=True, help="Target Azure Search index name")
    parser.add_argument("--out", required=False, default=None, help="Output CSV path for detailed rows")
    parser.add_argument("--summary-out", required=False, default=None, help="Output CSV path for summary counts (filename,count)")
    parser.add_argument("--unique-out", required=False, default=None, help="Output CSV path for unique filenames (one per row)")
    args = parser.parse_args()

    out_csv = args.out or os.path.join("outputs", f"{args.index}_filename_document_id.csv")
    summary_csv = args.summary_out or os.path.join("outputs", f"{args.index}_filename_counts.csv")
    unique_csv = args.unique_out or os.path.join("outputs", f"{args.index}_unique_filenames.csv")

    stats = export_filename_docid(args.index, out_csv=out_csv, summary_csv=summary_csv, unique_out_csv=unique_csv)

    print(f"Exported: {out_csv}")  # noqa: T201
    print(f"Summary: {summary_csv}")  # noqa: T201
    print(f"Unique: {unique_csv}")  # noqa: T201
    print(f"Rows: {stats['total_rows']}, Unique files: {stats['unique_filenames']}, Max chunks/file: {stats['max_chunks_per_file']}")  # noqa: T201


if __name__ == "__main__":
    main()