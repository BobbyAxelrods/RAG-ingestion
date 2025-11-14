import os
import sys
import json
import glob
import time
from typing import List, Dict, Any

import requests

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, total=None, desc=None, unit=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else range(total or 0)

# Load environment variables from a .env file if available (align with other scripts)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    # Local imports within the repo
    from src.search_index_prod.simple_index_config import build_simple_index_schema
    from src.search_index_prod.simple_uploader import load_etl_file, build_docs_from_etl
except Exception:
    # Fallback for execution contexts where module paths are different
    from simple_index_config import build_simple_index_schema  # type: ignore
    from simple_uploader import load_etl_file, build_docs_from_etl  # type: ignore


API_VERSION = os.getenv("AZURE_SEARCH_API_VERSION", "2024-07-01")


def create_or_update_index(endpoint: str, api_key: str, index_name: str, schema: Dict[str, Any]) -> None:
    url = f"{endpoint}/indexes/{index_name}?api-version={API_VERSION}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    resp = requests.put(url, headers=headers, data=json.dumps(schema))
    # Azure AI Search may return:
    # - 201 Created for new index
    # - 200 OK for upsert (depending on service behavior)
    # - 204 No Content for successful update with no body
    if resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"Index create/update failed: {resp.status_code} {resp.text}")


def upload_documents(endpoint: str, api_key: str, index_name: str, docs: List[Dict[str, Any]], batch_size: int = 1000) -> None:
    url = f"{endpoint}/indexes/{index_name}/docs/index?api-version={API_VERSION}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    pbar = tqdm(total=len(docs), desc="Uploading documents", unit="docs")
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        payload = {"value": [{"@search.action": "upload", **doc} for doc in batch]}
        resp = requests.post(url, headers=headers, data=json.dumps(payload))
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Upload batch failed: {resp.status_code} {resp.text}")
        # Update progress bar by the number of docs uploaded in this batch
        try:
            pbar.update(len(batch))
        except Exception:
            pass
        time.sleep(0.05)
    try:
        pbar.close()
    except Exception:
        pass


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create/update Azure AI Search index and upload ETL JSONs")
    parser.add_argument("--endpoint", type=str, default=os.getenv("AZURE_SEARCH_ENDPOINT") or os.getenv("SEARCH_SERVICE_ENDPOINT"), help="Azure AI Search endpoint, e.g. https://<service>.search.windows.net")
    parser.add_argument("--api-key", type=str, default=os.getenv("AZURE_SEARCH_API_KEY") or os.getenv("SEARCH_API_KEY") or os.getenv("SEARCH_SERVICE_KEY"), help="Azure AI Search admin API key")
    parser.add_argument("--index", type=str, default=os.getenv("SEARCH_INDEX_NAME") or os.getenv("AZURE_SEARCH_INDEX_NAME"), help="Target index name")
    parser.add_argument("--etl-glob", type=str, nargs="+", required=True, help="Glob(s) for ETL JSON files to ingest")
    parser.add_argument("--batch-size", type=int, default=1000, help="Upload batch size")
    parser.add_argument("--print-schema", action="store_true", help="Print the resolved schema and exit")

    args = parser.parse_args()

    # Require index for schema resolution
    if not args.index:
        raise SystemExit("Missing --index (or corresponding env var)")

    # Build schema and optionally print without requiring endpoint/api-key
    schema = build_simple_index_schema(args.index)
    if args.print_schema:
        print(json.dumps(schema, indent=2))
        return

    # For create/update and upload, endpoint and api-key are required
    if not args.endpoint or not args.api_key:
        raise SystemExit("Missing --endpoint or --api-key (or corresponding env vars)")

    # Create or update index
    create_or_update_index(args.endpoint, args.api_key, args.index, schema)

    # Resolve ETL files
    files: List[str] = []
    for g in args.etl_glob:
        files.extend(glob.glob(g))
    files = sorted(files)
    if not files:
        raise SystemExit("No ETL JSON files matched the provided glob(s)")

    # Build documents (BM25 fields always duplicated to preserve context)
    all_docs: List[Dict[str, Any]] = []
    for path in tqdm(files, desc="Processing ETL JSON files", unit="file"):
        try:
            etl_json = load_etl_file(path)
            docs = build_docs_from_etl(etl_json)
            all_docs.extend(docs)
        except Exception as e:
            print(f"Skipping {path} due to error: {e}")  # noqa: T201
            continue

    if not all_docs:
        raise SystemExit("No documents produced from ETL JSONs")

    print(f"Prepared {len(all_docs)} documents from {len(files)} ETL files. Starting upload...")

    # Upload
    upload_documents(args.endpoint, args.api_key, args.index, all_docs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()