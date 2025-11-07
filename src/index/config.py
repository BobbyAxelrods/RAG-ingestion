import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AzureSearchIndexingConfig:
    endpoint: str
    api_key: str
    index_name: str
    etl_schema_path: Optional[str] = None
    input_path: Optional[str] = None
    batch_size: int = 1000
    concurrency: int = 4
    dry_run: bool = False


def _read_env(*names: str) -> Optional[str]:
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return None


def load_config() -> AzureSearchIndexingConfig:
    endpoint = _read_env(
        "SEARCH_SERVICE_ENDPOINT",
        "AZURE_SEARCH_ENDPOINT",
        "AZUREAISEARCH_ENDPOINT",
    )
    api_key = _read_env(
        "SEARCH_SERVICE_KEY",
        "AZURE_SEARCH_KEY",
        "AZUREAISEARCH_KEY",
    )
    index_name = _read_env(
        "ETL_SEARCH_INDEX_NAME",
        "SEARCH_INDEX_NAME",
        "AZURE_SEARCH_INDEX_NAME",
    ) or "insurance-documents-index"
    etl_schema_path = _read_env("INDEX_SCHEMA_PATH")
    input_path = _read_env("INDEX_INPUT_PATH")
    batch_size = int(os.getenv("INDEX_BATCH_SIZE", "1000"))
    concurrency = int(os.getenv("INDEX_CONCURRENCY", "4"))
    dry_run = os.getenv("INDEX_DRY_RUN", "false").lower() in ("1", "true", "yes")

    if not endpoint or not api_key:
        raise RuntimeError(
            "Azure Search endpoint and key must be set via env: SEARCH_SERVICE_ENDPOINT and SEARCH_SERVICE_KEY"
        )

    return AzureSearchIndexingConfig(
        endpoint=endpoint,
        api_key=api_key,
        index_name=index_name,
        etl_schema_path=etl_schema_path,
        input_path=input_path,
        batch_size=batch_size,
        concurrency=concurrency,
        dry_run=dry_run,
    )