import os
import argparse
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient


def _get_env(*names: str, required: bool = False, default: str | None = None) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    if required and default is None:
        raise RuntimeError(f"Missing env var: one of {', '.join(names)}")
    return default


def build_index_client() -> SearchIndexClient:
    endpoint = _get_env("SEARCH_SERVICE_ENDPOINT", "AZURE_SEARCH_ENDPOINT", required=True)
    api_key = _get_env("SEARCH_SERVICE_KEY", "AZURE_SEARCH_API_KEY", required=True)
    return SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))


def list_indexes(include_stats: bool = False) -> list[dict]:
    client = build_index_client()
    indexes = []
    for idx in client.list_indexes():
        item = {"name": getattr(idx, "name", None)}
        if include_stats:
            try:
                stats = client.get_index_statistics(item["name"])  # type: ignore
                item["document_count"] = getattr(stats, "document_count", None)
                item["storage_size"] = getattr(stats, "storage_size", None)
            except Exception:
                pass
        indexes.append(item)
    return indexes


def main():
    parser = argparse.ArgumentParser(description="List Azure AI Search index names")
    parser.add_argument("--out", default=None, help="Optional path to write JSON output")
    parser.add_argument("--stats", action="store_true", help="Include document and storage stats per index")
    args = parser.parse_args()

    data = list_indexes(include_stats=args.stats)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(data)} indexes to {args.out}")
    else:
        for d in data:
            line = d["name"]
            if args.stats:
                line += f" (docs={d.get('document_count')}, size={d.get('storage_size')})"
            print(line)


if __name__ == "__main__":
    main()