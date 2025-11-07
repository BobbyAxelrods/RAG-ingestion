import os
from typing import Any, Dict, List, Optional, Tuple

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from src.config import AzureSearchConfig


class SearchEngine:
    """Wraps Azure AI Search queries for different strategies."""

    def __init__(self, endpoint: Optional[str] = None, index_name: Optional[str] = None, api_key: Optional[str] = None):
        # Load from AzureSearchConfig directly to avoid global Config side-effects (OFFLINE_MODE prints, metadata warnings)
        cfg = AzureSearchConfig()
        endpoint = endpoint or os.getenv("AZURE_SEARCH_ENDPOINT") or cfg.endpoint
        api_key = api_key or os.getenv("AZURE_SEARCH_API_KEY") or cfg.key
        index_name = index_name or os.getenv("AZURE_SEARCH_INDEX_NAME") or cfg.index_name
        if not endpoint or not index_name or not api_key:
            raise RuntimeError("Missing Azure Search configuration: endpoint/index_name/api_key")
        self.index_name = index_name
        self.client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))

    def _run(self, query: str, top_k: int, search_fields: Optional[List[str]] = None, filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        results = self.client.search(search_text=query, top=top_k, filter=filter_expr, search_fields=search_fields)
        out: List[Dict[str, Any]] = []
        for r in results:
            doc: Dict[str, Any] = {
                "file_name": r.get("file_name") or r.get("sys_file_name"),
                "sys_file_name": r.get("sys_file_name"),
                "chunk_content": r.get("chunk_content"),
                "chunk_page_number": r.get("chunk_page_number"),
                "qa_answers": r.get("qa_answers"),
                "qa_confidence": r.get("qa_confidence"),
                "score": getattr(r, "score", None),
            }
            out.append(doc)
        return out

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        fields = [
            "chunk_content",
            "qa_questions",
            "chunk_entities",
            "chunk_function_summary",
            "file_name",
        ]
        return self._run(query, top_k, search_fields=fields)

    def qa_pairs(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self._run(query, top_k, search_fields=["qa_questions", "qa_answers"])

    def entity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self._run(query, top_k, search_fields=["chunk_entities"])

    def document_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        fields = ["file_name", "sys_file_name", "title_name_en", "title_name_tc"]
        return self._run(query, top_k, search_fields=fields)

    def summary_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        fields = ["chunk_function_summary", "chunk_content"]
        return self._run(query, top_k, search_fields=fields)

    def run_strategy(self, strategy: str, query: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        s = strategy.upper()
        if s == "QA_PAIRS":
            return s, self.qa_pairs(query, top_k)
        if s == "ENTITY_SEARCH":
            return s, self.entity_search(query, top_k)
        if s == "DOCUMENT_SEARCH":
            return s, self.document_search(query, top_k)
        if s == "SUMMARY_SEARCH":
            return s, self.summary_search(query, top_k)
        return "HYBRID_SEARCH", self.hybrid_search(query, top_k)