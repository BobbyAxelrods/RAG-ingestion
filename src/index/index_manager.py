import json
import os
from typing import Optional

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
import requests


def _normalize_schema_dict(data: dict) -> dict:
    # Normalize vector field property name for current API
    fields = data.get("fields", [])
    for field in fields:
        if field.get("type") == "Collection(Edm.Single)":
            # For vector fields, ensure 'vectorSearchConfiguration' is set
            profile = field.pop("vectorSearchProfile", None)
            if profile and not field.get("vectorSearchConfiguration"):
                field["vectorSearchConfiguration"] = profile
            # Ensure dimensions present
            if not field.get("dimensions"):
                # fallback to 1536 if unspecified
                field["dimensions"] = 1536
    data["fields"] = fields
    # No-op normalization for vectorSearch section; rely on provided schema.
    return data


def _load_schema(schema_path: str) -> SearchIndex:
    with open(schema_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    data = _normalize_schema_dict(raw)
    return SearchIndex(**data)


class IndexManager:
    def __init__(self, endpoint: str, api_key: str):
        self.client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))

    def exists(self, index_name: str) -> bool:
        try:
            _ = self.client.get_index(index_name)
            return True
        except ResourceNotFoundError:
            return False

    def ensure_index(self, index_name: str, schema_path: Optional[str] = None) -> None:
        if self.exists(index_name):
            return
        if not schema_path:
            raise RuntimeError(
                "Index does not exist and no schema_path provided to create it."
            )
        # Try REST creation to avoid SDK model mismatches for vector configs
        with open(schema_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        data = _normalize_schema_dict(raw)
        data["name"] = index_name
        api_version = os.getenv("AZURE_SEARCH_API_VERSION") or os.getenv("CSV_AI_INDEX_API_VERSION") or "2024-08-01-Preview"
        url = self.client._endpoint.rstrip("/") + f"/indexes/{index_name}?api-version={api_version}"
        headers = {"Content-Type": "application/json", "api-key": self.client._credential.key}
        resp = requests.put(url, headers=headers, data=json.dumps(data))
        if resp.status_code >= 300:
            raise RuntimeError(f"Failed to create index: {resp.status_code} {resp.text}")