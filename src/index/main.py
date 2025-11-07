from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List

from dotenv import load_dotenv

from .config import load_config
from .discovery import discover_json_files
from .transformers.etl_flatten import flatten_etl_json
from .uploader import BatchUploader
from .flat_index_service import FlatIndexService


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Index flattened ETL JSON outputs into Azure AI Search (flat schema)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to ETL JSON file or directory containing ETL JSON files",
    )
    parser.add_argument(
        "--pattern",
        default="*_extraction.json",
        help="Glob suffix/pattern when --input is a directory (default: *_extraction.json)",
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Create the flat index if it does not exist using --schema or env",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Path to index schema JSON (overrides env INDEX_SCHEMA_PATH)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Upload batch size (default from env INDEX_BATCH_SIZE)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not upload, only prepare and print counts",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Run a sample text query against the index (top 5)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("index.main")

    load_dotenv()
    cfg = load_config()

    schema_path = args.schema or cfg.etl_schema_path or os.path.join(
        os.path.dirname(__file__), "..", "..", "project_plan", "schema", "aisearch_index_schema.json"
    )

    # Ensure index exists (SDK-only)
    if args.create_index:
        # Read OpenAI settings directly from environment to optionally enable on-service vectorization
        openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("OPENAI_ENDPOINT")
        openai_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        openai_deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or os.environ.get("OPENAI_EMBEDDING_DEPLOYMENT")
        try:
            embedding_dimensions = int(os.environ.get("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536"))
        except Exception:
            embedding_dimensions = 1536

        flat_service = FlatIndexService(
            endpoint=cfg.endpoint,
            api_key=cfg.api_key,
            index_name=cfg.index_name,
            openai_endpoint=openai_endpoint,
            openai_key=openai_key,
            openai_deployment=openai_deployment,
            embedding_dimensions=embedding_dimensions,
        )
        ok = flat_service.create_index(recreate=False)
        if ok:
            logger.info("Index ensured via SDK-only creation.")
        else:
            raise RuntimeError("Failed to create index via SDK.")

    # Collect files
    target = args.input
    files: List[str] = []
    if os.path.isfile(target):
        files = [target]
    else:
        files = discover_json_files(target, pattern_suffix=args.pattern)
    if not files:
        logger.warning("No ETL JSON files found for indexing.")
        return

    # Prepare docs
    total_docs = 0
    all_docs: List[dict] = []
    for fp in files:
        data = _load_json(fp)
        flat_docs = flatten_etl_json(data)
        total_docs += len(flat_docs)
        all_docs.extend(flat_docs)
        logger.info(f"Prepared {len(flat_docs)} docs from {os.path.basename(fp)}")

    logger.info(f"Total prepared docs: {total_docs}")
    if args.dry_run or cfg.dry_run:
        logger.info("Dry run enabled; skipping upload.")
    else:
        uploader = BatchUploader(endpoint=cfg.endpoint, api_key=cfg.api_key, index_name=cfg.index_name)
        batch_size = args.batch or cfg.batch_size
        uploaded = uploader.upload(all_docs, batch_size=batch_size)
        logger.info(f"Uploaded {uploaded} docs to index '{cfg.index_name}'")

    # Optional text query
    if args.query:
        try:
            from azure.search.documents import SearchClient
            from azure.core.credentials import AzureKeyCredential

            search_client = SearchClient(endpoint=cfg.endpoint, index_name=cfg.index_name, credential=AzureKeyCredential(cfg.api_key))
            results = search_client.search(search_text=args.query, top=5, query_type="simple")
            logger.info("Top 5 search results:")
            for i, r in enumerate(results):
                content = r.get("chunk_content", "")
                page = r.get("chunk_page_number", None)
                file_name = r.get("file_name", r.get("sys_file_name", ""))
                logger.info(f"[{i+1}] page={page} file={file_name} content={content[:200].replace('\n',' ')}")
        except Exception as e:
            logger.error(f"Failed to run query: {e}")


if __name__ == "__main__":
    main()