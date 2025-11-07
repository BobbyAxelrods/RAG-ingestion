import logging
from typing import Any, List, Optional

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from src.etl.config import AzureSearchConfig
from src.etl.services.openai_service import OpenAIService
from src.config import get_config

logger = logging.getLogger(__name__)


class SearchStrategyEngine:
    """
    Implements key search strategies against Azure AI Search.
    """

    def __init__(self, index_name: Optional[str] = None):
        cfg = get_config()
        self.search_cfg: AzureSearchConfig = AzureSearchConfig(index_name=index_name or cfg.azure_search.index_name)
        self.credential = AzureKeyCredential(self.search_cfg.key)
        self.client = SearchClient(
            endpoint=self.search_cfg.endpoint,
            index_name=self.search_cfg.index_name,
            credential=self.credential,
        )
        # Embeddings (optional for hybrid)
        try:
            self.openai = OpenAIService(cfg.azure_openai)
        except Exception:
            self.openai = None

    def _build_filters(self, context: dict, min_word_count: Optional[int] = None) -> Optional[str]:
        filters: List[str] = []
        branch = context.get("branch")
        category = context.get("category")
        library = context.get("library")
        if branch:
            filters.append(f"branch_name eq '{branch}'")
        if category:
            filters.append(f"category_name_en eq '{category}'")
        if library:
            filters.append(f"library_name_en eq '{library}'")
        if isinstance(min_word_count, int):
            filters.append(f"chunk_word_count ge {min_word_count}")
        return " and ".join(filters) if filters else None

    def hybrid_search(self, query: str, context: dict, top_k: int = 10) -> List[dict]:
        embedding = None
        if self.openai:
            try:
                embedding = self.openai.generate_embedding(query)
            except Exception:
                embedding = None
        filter_str = self._build_filters(context, min_word_count=50)
        kwargs: dict[str, Any] = {
            "search_text": query,
            "filter": filter_str,
            "select": "doc_id,file_name,chunk_content,chunk_page_number,chunk_function_summary,file_url,qa_questions,qa_answers",
            "top": top_k,
        }
        if embedding:
            kwargs["vector_queries"] = [{
                "kind": "vector",
                "vector": embedding,
                "k_nearest_neighbors": 50,
                "fields": "chunk_content_vector",
            }]
        results = self.client.search(**kwargs)
        return list(results)

    def qa_search(self, query: str, context: dict, top_k: int = 5, min_confidence: float = 0.0) -> List[dict]:
        filters: List[str] = []
        filters.append(f"qa_confidence ge {min_confidence}")
        branch = context.get("branch")
        if branch:
            filters.append(f"branch_name eq '{branch}'")
        filter_str = " and ".join(filters)
        results = self.client.search(
            search_text=query,
            search_fields=["qa_questions", "qa_answers"],
            filter=filter_str,
            select="qa_questions,qa_answers,file_name,chunk_page_number,qa_confidence",
            top=top_k,
        )
        return list(results)

    def entity_search(self, query: str, entity: Optional[str], context: dict, top_k: int = 10) -> List[dict]:
        filters: List[str] = []
        if entity:
            filters.append(f"chunk_entities/any(e: e eq '{entity}')")
        branch = context.get("branch")
        if branch:
            filters.append(f"branch_name eq '{branch}'")
        filter_str = " and ".join(filters) if filters else None
        results = self.client.search(
            search_text=query,
            filter=filter_str,
            select="file_name,chunk_page_number,chunk_entities,chunk_function_summary",
            facets=["file_name,count:10"],
            top=top_k,
        )
        return list(results)

    def summary_search(self, query: str, context: dict, top_k: int = 10) -> List[dict]:
        filter_str = self._build_filters(context)
        results = self.client.search(
            search_text=query,
            search_fields=["chunk_function_summary"],
            filter=filter_str,
            select="file_name,chunk_page_number,chunk_function_summary",
            top=top_k,
        )
        return list(results)

    def document_search(self, query: str, filename: str, start_page: Optional[int] = None, end_page: Optional[int] = None, top_k: int = 100) -> List[dict]:
        filters: List[str] = [f"file_name eq '{filename}'"]
        if isinstance(start_page, int):
            filters.append(f"chunk_page_number ge {start_page}")
        if isinstance(end_page, int):
            filters.append(f"chunk_page_number le {end_page}")
        filter_str = " and ".join(filters)
        results = self.client.search(
            search_text=query,
            filter=filter_str,
            select="chunk_page_number,chunk_content,chunk_function_summary,file_url,qa_questions,qa_answers",
            top=top_k,
        )
        return list(results)