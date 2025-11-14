import os
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
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


def delete_index(index_name: str) -> None:
    client = build_index_client()
    try:
        client.delete_index(index_name)
        print(f"Deleted index: {index_name}")
    except HttpResponseError as e:
        print(f"Failed to delete index '{index_name}': {e.message}")
    except Exception as e:
        print(f"Error deleting index '{index_name}': {e}")


def main():
    parser = argparse.ArgumentParser(description="Delete an Azure AI Search index by name")
    parser.add_argument("index", help="Index name to delete")
    parser.add_argument("--yes", action="store_true", help="Confirm deletion (non-interactive)")
    args = parser.parse_args()

    if not args.yes:
        print("Refusing to delete without --yes confirmation. Aborting.")
        return

    delete_index(args.index)


if __name__ == "__main__":
    main()