import os
import json
import sys
import argparse
import requests

try:
    from tqdm import tqdm  # progress bar
except Exception:
    def tqdm(it, **kwargs):
        return it  # graceful fallback if tqdm is not available


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _discover_json_files(path: str, suffix: str = ".json") -> list[str]:
    files: list[str] = []
    if os.path.isfile(path):
        if path.endswith(".json"):
            files.append(path)
        return files
    for root, _, filenames in os.walk(path):
        for name in filenames:
            if name.endswith(suffix):
                files.append(os.path.join(root, name))
    return files


def create_or_update_index(schema_path: str, index_name: str | None = None, api_version: str = "2024-07-01") -> bool:
    load_env()
    endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT") or os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("SEARCH_SERVICE_KEY") or os.getenv("AZURE_SEARCH_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError("Missing SEARCH_SERVICE_ENDPOINT or SEARCH_SERVICE_KEY in environment")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Sanitize unsupported properties for Azure Search REST API
    if "$schema" in schema:
        schema.pop("$schema", None)

    def strip_comments(obj):
        if isinstance(obj, dict):
            if "comment" in obj:
                obj.pop("comment", None)
            for k in list(obj.keys()):
                strip_comments(obj[k])
        elif isinstance(obj, list):
            for item in obj:
                strip_comments(item)

    strip_comments(schema)

    # Additional sanitation: remove analyzer on non-searchable fields
    fields = schema.get("fields") or []
    if isinstance(fields, list):
        for fld in fields:
            if isinstance(fld, dict):
                searchable = fld.get("searchable", None)
                if (searchable is False or searchable == "false") and "analyzer" in fld:
                    fld.pop("analyzer", None)

        # Ensure key field is retrievable (Azure requirement)
        for fld in fields:
            if isinstance(fld, dict) and fld.get("key"):
                # If retrievable is explicitly false, set to true
                if str(fld.get("retrievable", "true")).lower() in ("false", "0"):
                    fld["retrievable"] = True

    # Remove unsupported suggesters block if present
    if "suggesters" in schema:
        schema.pop("suggesters", None)

    # Ensure index name in body matches desired name
    desired_name = (index_name or schema.get("name") or "experiment_1").strip()
    schema["name"] = desired_name

    # Create or update index via REST
    url = endpoint.rstrip("/") + f"/indexes/{desired_name}?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }

    resp = requests.put(url, headers=headers, data=json.dumps(schema))
    if resp.status_code in (200, 201):
        print(f"Index '{desired_name}' created/updated successfully.")
        return True
    elif resp.status_code == 409:
        print(f"Index '{desired_name}' already exists. Proceeding with incremental upload.")
        return True
    else:
        print(f"Failed to create index: {resp.status_code}\n{resp.text}")
        return False


def index_directory(input_dir: str, index_name: str, batch_size: int = 1000, suffix: str = ".json") -> int:
    """Index all ETL JSON documents under a directory into Azure AI Search with tqdm progress."""
    load_env()
    # Lazy imports to avoid circular when script run as module or standalone
    try:
        from .transformers.etl_flatten import flatten_etl_json
    except Exception:
        # Fallback when running as a plain script
        from transformers.etl_flatten import flatten_etl_json  # type: ignore
    try:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
    except Exception as e:
        raise RuntimeError(f"Azure Search SDK not available: {e}")

    endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT") or os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("SEARCH_SERVICE_KEY") or os.getenv("AZURE_SEARCH_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError("Missing SEARCH_SERVICE_ENDPOINT or SEARCH_SERVICE_KEY in environment")

    files = _discover_json_files(input_dir, suffix=suffix)
    if not files:
        print("No JSON files found to index.")
        return 0

    client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))

    uploaded = 0
    buffer: list[dict] = []

    print(f"Found {len(files)} JSON file(s). Starting ingestion...")
    for fp in tqdm(files, desc="Files", unit="file"):
        try:
            data = _load_json(fp)
        except Exception as e:
            print(f"Failed to read {fp}: {e}")
            continue

        try:
            flat_docs = flatten_etl_json(data)
        except Exception as e:
            print(f"Failed to flatten {fp}: {e}")
            continue

        for doc in tqdm(flat_docs, desc="Documents", unit="doc", leave=False):
            buffer.append(doc)
            if len(buffer) >= batch_size:
                try:
                    results = client.upload_documents(documents=buffer)
                    uploaded += len(results)
                except Exception as e:
                    print(f"Upload batch failed: {e}")
                finally:
                    buffer.clear()

    if buffer:
        try:
            results = client.upload_documents(documents=buffer)
            uploaded += len(results)
        except Exception as e:
            print(f"Final upload batch failed: {e}")
        finally:
            buffer.clear()

    print(f"Indexed {uploaded} document(s) to '{index_name}'.")
    return uploaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create/Update Azure AI Search index and bulk index ETL JSONs with progress")
    parser.add_argument("--schema", type=str, default=None, help="Path to index schema JSON")
    parser.add_argument("--index", type=str, default=None, help="Azure Search index name (overrides schema name)")
    parser.add_argument("--api-version", type=str, default=os.getenv("SEARCH_API_VERSION") or "2024-07-01", help="API version for index creation")
    parser.add_argument("--input", type=str, default=None, help="Path to ETL JSON file or directory to ingest")
    parser.add_argument("--suffix", type=str, default=".json", help="Filename suffix to match when scanning directories (default: .json)")
    parser.add_argument("--batch", type=int, default=1000, help="Upload batch size (default: 1000)")

    args = parser.parse_args()

    # If schema is provided, create/update the index first
    if args.schema:
        ok = create_or_update_index(args.schema, args.index, api_version=args.api_version)
        if not ok:
            sys.exit(1)

    # If input provided, index documents
    if args.input and (args.index or (args.schema is not None)):
        index_name = args.index
        if not index_name and args.schema:
            # Read the schema name if index not specified
            try:
                schema_obj = _load_json(args.schema)
                index_name = schema_obj.get("name") or "experiment_1"
            except Exception:
                index_name = "experiment_1"
        count = index_directory(args.input, index_name=index_name, batch_size=args.batch, suffix=args.suffix)
        sys.exit(0 if count >= 0 else 1)

    # Backward compatibility: if only positional args were provided previously
    if not args.schema and not args.input:
        print("Nothing to do. Provide --schema to create/update index and/or --input to ingest documents.")
        sys.exit(2)