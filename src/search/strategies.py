from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from azure.search.documents.models import VectorizedQuery, QueryCaptionType, QueryAnswerType


class BaseRAGStrategy:
    """Common interface for RAG strategies.

    Each strategy must implement generate_answer returning a dict:
    {"query", "answer", "sources", "retrieval_method", "model", "token_usage"}
    """

    def __init__(self, search_client, openai_client, embedding_function=None):
        self.search_client = search_client
        self.openai_client = openai_client
        self.embedding_function = embedding_function

    def _is_traditional_chinese(self, text: str) -> bool:
        return bool(re.search(r"[\u4E00-\u9FFF]", text))

    def _build_llm_answer(self, query: str, context: str, model: str, temperature: float = 0.2, max_tokens: int = 700) -> Dict[str, Any]:
        system_message = (
            "You are a helpful assistant that answers questions based on the provided context. "
            "Always cite your sources using [Source N]. "
            "If the context lacks sufficient information, state that clearly."
        )

        user_message = f"""Context:
{context}

Question: {query}

Provide a comprehensive answer based on the context, and cite sources."""

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }


class HybridRAGStrategy(BaseRAGStrategy):
    """Hybrid RAG using BM25 + vector aligned to index schema."""

    def generate_answer(
        self,
        *,
        query: str,
        top_k: int = 5,
        k_neighbors: int = 50,
        model: str,
        document_id_filter: Optional[str] = None,
        branch_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Build vector query
        query_vector: List[float] = self.embedding_function(query)
        vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=k_neighbors, fields="chunk_content_vector")

        # Optional filters
        filters: List[str] = []
        if document_id_filter:
            filters.append(f"document_id eq '{document_id_filter}'")
        if branch_filter:
            filters.append(f"branch_name eq '{branch_filter}'")
        filter_expr = " and ".join(filters) if filters else None

        # Semantic config by language
        semantic_config = (
            "insurance-semantic-config-tc" if self._is_traditional_chinese(query) else "insurance-semantic-config-en"
        )

        results = list(
            self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                top=top_k,
                select=[
                    "doc_id",
                    "file_name",
                    "chunk_content",
                    "file_url",
                    "chunk_page_number",
                    "document_id",
                    "library_name_en",
                    "library_name_tc",
                    "category_name_en",
                    "category_name_tc",
                ],
                filter=filter_expr,
                include_total_count=False,
                semantic_configuration_name=semantic_config,
            )
        )

        # Build context and sources
        context_parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        for rank, r in enumerate(results, 1):
            fn = r.get("file_name")
            pg = r.get("chunk_page_number")
            content = r.get("chunk_content")
            context_parts.append(f"[Source {rank}] {fn} (page {pg})\n{content}\n")
            sources.append(
                {
                    "rank": rank,
                    "doc_id": r.get("doc_id"),
                    "file_name": fn,
                    "page": pg,
                    "url": r.get("file_url"),
                    "document_id": r.get("document_id"),
                    "library_name_en": r.get("library_name_en"),
                    "library_name_tc": r.get("library_name_tc"),
                    "category_name_en": r.get("category_name_en"),
                    "category_name_tc": r.get("category_name_tc"),
                    "search_score": r.get("@search.score"),
                }
            )
        context = "\n".join(context_parts)

        llm = self._build_llm_answer(query, context, model)
        return {
            "query": query,
            "answer": llm["content"],
            "sources": sources,
            "retrieval_method": "hybrid",
            "model": model,
            "token_usage": llm["usage"],
        }


class HierarchicalRAGStrategy(BaseRAGStrategy):
    """Three-stage hierarchical retrieval then answer generation."""

    def __init__(
        self,
        search_client,
        openai_client,
        embedding_function,
        hierarchical_searcher_cls,
    ):
        super().__init__(search_client, openai_client, embedding_function)
        self.hierarchical_searcher_cls = hierarchical_searcher_cls

    def generate_answer(
        self,
        *,
        query: str,
        top_k: int = 5,
        k_neighbors: int = 50,
        model: str,
        document_id_filter: Optional[str] = None,
        branch_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Configure searcher
        searcher = self.hierarchical_searcher_cls(
            search_client=self.search_client,
            openai_client=self.openai_client,
            bm25_search_fields=["chunk_content"],
            vector_fields=["chunk_content_vector"],
            select_fields=[
                "doc_id",
                "file_name",
                "chunk_content",
                "file_url",
                "chunk_page_number",
                "document_id",
            ],
            id_field="doc_id",
            candidate_filter_field="file_name",
            max_candidate_filter_values=100,
            use_semantic_reranking=False,
        )

        # External filter
        fexpr: Optional[str] = None
        filt: List[str] = []
        if document_id_filter:
            filt.append(f"document_id eq '{document_id_filter}'")
        if branch_filter:
            filt.append(f"branch_name eq '{branch_filter}'")
        if filt:
            fexpr = " and ".join(filt)

        # Retrieve final results
        results = searcher.search(query=query, top=top_k, filter_expression=fexpr, stage2_candidates=k_neighbors)

        # Build context and sources
        context_parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        for rank, r in enumerate(results, 1):
            fn = r.get("file_name") if isinstance(r, dict) else getattr(r, "file_name", None)
            pg = r.get("chunk_page_number") if isinstance(r, dict) else getattr(r, "chunk_page_number", None)
            content = r.get("chunk_content") if isinstance(r, dict) else getattr(r, "chunk_content", None)
            doc_id = r.get("doc_id") if isinstance(r, dict) else getattr(r, "doc_id", None)
            url = r.get("file_url") if isinstance(r, dict) else getattr(r, "file_url", None)
            context_parts.append(f"[Source {rank}] {fn} (page {pg})\n{content}\n")
            sources.append(
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "file_name": fn,
                    "page": pg,
                    "url": url,
                    "search_score": r.get("@search.score") if isinstance(r, dict) else getattr(r, "@search.score", None),
                    "timings": r.get("_timings") if isinstance(r, dict) else getattr(r, "_timings", None),
                }
            )
        context = "\n".join(context_parts)

        llm = self._build_llm_answer(query, context, model)
        return {
            "query": query,
            "answer": llm["content"],
            "sources": sources,
            "retrieval_method": "hierarchical_hybrid",
            "model": model,
            "token_usage": llm["usage"],
        }


class SemanticRAGStrategy(BaseRAGStrategy):
    """Semantic search + captions/answers with LLM generation."""

    def generate_answer(
        self,
        *,
        query: str,
        top_k: int = 5,
        model: str,
        semantic_config_tc: str = "insurance-semantic-config-tc",
        semantic_config_en: str = "insurance-semantic-config-en",
    ) -> Dict[str, Any]:
        semantic_config = semantic_config_tc if self._is_traditional_chinese(query) else semantic_config_en

        results = list(
            self.search_client.search(
                search_text=query,
                query_type="semantic",
                semantic_configuration_name=semantic_config,
                query_caption=QueryCaptionType.EXTRACTIVE,
                query_answer=QueryAnswerType.EXTRACTIVE,
                select=[
                    "doc_id",
                    "file_name",
                    "chunk_content",
                    "file_url",
                    "chunk_page_number",
                ],
                top=top_k,
            )
        )

        # Build context preferring semantic captions
        context_parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        for rank, r in enumerate(results, 1):
            captions = r.get("@search.captions") or []
            cap_text = captions[0].get("text") if captions else r.get("chunk_content")
            fn = r.get("file_name")
            pg = r.get("chunk_page_number")
            context_parts.append(f"[Source {rank}] {fn} (page {pg})\n{cap_text}\n")
            sources.append(
                {
                    "rank": rank,
                    "doc_id": r.get("doc_id"),
                    "file_name": fn,
                    "page": pg,
                    "url": r.get("file_url"),
                    "search_score": r.get("@search.score"),
                    "rerank_score": r.get("@search.rerankScore"),
                }
            )
        context = "\n".join(context_parts)

        # Prefer high-confidence semantic answers if present
        best_answer: Optional[Dict[str, Any]] = None
        for r in results:
            answers = r.get("@search.answers") or []
            if answers:
                ba = max(answers, key=lambda a: a.get("score", 0))
                if ba.get("score", 0) > 0.85:
                    best_answer = {"content": ba.get("text", ""), "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
                    break

        llm = best_answer or self._build_llm_answer(query, context, model)
        return {
            "query": query,
            "answer": llm["content"],
            "sources": sources,
            "retrieval_method": "semantic",
            "model": model,
            "token_usage": llm["usage"],
        }