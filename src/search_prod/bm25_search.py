import os
import logging
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
import argparse
import json
from dotenv import load_dotenv

load_dotenv()


class AzureBM25Search:
    """
    BM25 search client aligned to your Azure AI Search index schema.

    - Uses env-configured endpoint/key/index.
    - Selects fields that exist: file_name, chunk_content, chunk_page_number, etc.
    - Supports simple filters (equality, gt/lt) with OData.
    """

    def __init__(self, endpoint: str, index_name: str, api_key: str):
        self.client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))
        self.logger = logging.getLogger(__name__)

    def search(self, query: str, top_k: int = 10) -> list:
        """BM25-only search over chunk content and file name."""
        try:
            results = self.client.search(
                search_text=query,
                top=top_k,
                query_type="simple",
                search_fields=["chunk_content"],
                select=[
                    "doc_id",
                    "file_name",
                    "chunk_content",
                    "file_url",
                    "chunk_page_number",
                    "document_id",
                    "branch_name",
                ],
            )
            return [
                {
                    "doc_id": r.get("doc_id"),
                    "file_name": r.get("file_name"),
                    "content": r.get("chunk_content"),
                    "page": r.get("chunk_page_number"),
                    "url": r.get("file_url"),
                    "document_id": r.get("document_id"),
                    "branch_name": r.get("branch_name"),
                    "score": r.get("@search.score"),
                }
                for r in results
            ]
        except HttpResponseError as e:
            self.logger.error(f"Search failed: {e.message}")
            raise

    def search_with_filters(self, query: str, filters: dict, top_k: int = 10) -> list:
        """BM25 search with additional OData filters (e.g., document_id, branch_name)."""
        filter_expression = self._build_filter_expression(filters)
        try:
            results = self.client.search(
                search_text=query,
                filter=filter_expression,
                top=top_k,
                query_type="simple",
                search_fields=["chunk_content"],
                select=[
                    "doc_id",
                    "file_name",
                    "chunk_content",
                    "file_url",
                    "chunk_page_number",
                    "document_id",
                    "branch_name",
                ],
            )
            return [
                {
                    "doc_id": r.get("doc_id"),
                    "file_name": r.get("file_name"),
                    "content": r.get("chunk_content"),
                    "page": r.get("chunk_page_number"),
                    "url": r.get("file_url"),
                    "document_id": r.get("document_id"),
                    "branch_name": r.get("branch_name"),
                    "score": r.get("@search.score"),
                }
                for r in results
            ]
        except HttpResponseError as e:
            self.logger.error(f"Filtered search failed: {e.message}")
            raise

    def search_with_facets(self, query: str, facet_fields: list, top_k: int = 10) -> dict:
        """BM25 search with facets to inspect distribution."""
        try:
            results = self.client.search(search_text=query, top=top_k, facets=facet_fields)
            documents = []
            facets = {}
            for r in results:
                documents.append(
                    {
                        "doc_id": r.get("doc_id"),
                        "file_name": r.get("file_name"),
                        "content": r.get("chunk_content"),
                        "page": r.get("chunk_page_number"),
                        "score": r.get("@search.score"),
                    }
                )
            if hasattr(results, "get_facets"):
                facet_results = results.get_facets()
                for facet_field in facet_fields:
                    if facet_field in facet_results:
                        facets[facet_field] = [
                            {"value": f["value"], "count": f["count"]} for f in facet_results[facet_field]
                        ]
            return {"documents": documents, "facets": facets, "total_count": len(documents)}
        except HttpResponseError as e:
            self.logger.error(f"Faceted search failed: {e.message}")
            raise

    def _build_filter_expression(self, filters: dict) -> str:
        """Build OData filter expression from dict (handles eq, gt, lt)."""
        parts = []
        for field, value in filters.items():
            if isinstance(value, str) and value.startswith(">"):
                comp_value = value.split()[1]
                parts.append(f"{field} gt {comp_value}")
            elif isinstance(value, str) and value.startswith("<"):
                comp_value = value.split()[1]
                parts.append(f"{field} lt {comp_value}")
            else:
                safe = str(value).replace("'", "''")
                parts.append(f"{field} eq '{safe}'")
        return " and ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="BM25 search against Azure AI Search index (env-configured)")
    parser.add_argument("--query", type=str, required=True, help="Query string")
    parser.add_argument("--top-k", type=int, default=6, help="Number of results")
    parser.add_argument("--document-id", type=str, default=None, help="Optional document_id filter")
    parser.add_argument("--branch", type=str, default=None, help="Optional branch_name filter")
    args = parser.parse_args()

    endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT") or os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("SEARCH_SERVICE_KEY") or os.getenv("AZURE_SEARCH_API_KEY")
    index_name = os.getenv("SEARCH_INDEX_NAME") or os.getenv("AZURE_SEARCH_INDEX_NAME")
    if not endpoint or not api_key or not index_name:
        raise RuntimeError("Missing SEARCH_SERVICE_ENDPOINT/KEY/INDEX_NAME in environment")

    client = AzureBM25Search(endpoint, index_name, api_key)

    filters = {}
    if args.document_id:
        filters["document_id"] = args.document_id
    if args.branch:
        filters["branch_name"] = args.branch

    results = client.search_with_filters(args.query, filters, top_k=args.top_k) if filters else client.search(args.query, top_k=args.top_k)

    print("Query:", args.query)  # noqa: T201
    print("\nResults:")  # noqa: T201
    result = []
    for i, r in enumerate(results, 1):
        fn = r.get("file_name")
        c = (r.get("content") or "")
        pg = r.get("page")
        sc = r.get("score")
        print(f"  {i}. {fn} (page {pg}) score={sc}")  # noqa: T201

        result.append({
            "rank": i,
            "file_name": fn,
            "page": pg,
            "score": sc,
            "document_id": r.get("document_id"),
            "branch_name": r.get("branch_name"),
            "url": r.get("url"),
            "content": c,
        })

    payload = {"query": args.query, "results": result}
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)




if __name__ == "__main__":
    main()