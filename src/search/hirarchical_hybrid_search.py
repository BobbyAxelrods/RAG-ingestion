from typing import List, Dict, Any, Optional
import time
import re

from azure.search.documents.models import VectorizedQuery


class HierarchicalHybridSearcher:
    """
    Three-stage hierarchical search for large corpora using Azure Cognitive Search:
    - Stage 1: BM25-only coarse retrieval to gather candidate IDs
    - Stage 2: Hybrid (BM25 + vector) within the candidate set
    - Stage 3: Optional semantic re-ranking for final ordering

    This implementation is index-aware and configurable for field names.
    """

    def __init__(
        self,
        search_client,
        openai_client,
        *,
        deployment_name: str = "text-embedding-small-3",
        id_field: str = "doc_id",
        candidate_filter_field: str = "file_name",
        vector_fields: Optional[List[str]] = None,
        bm25_search_fields: Optional[List[str]] = None,
        query_language: Optional[str] = None,
        select_fields: Optional[List[str]] = None,
        use_semantic_reranking: bool = False,
        semantic_configuration_name: Optional[str] = None,
        max_candidate_filter_values: int = 50,
    ) -> None:
        self.search_client = search_client
        self.openai_client = openai_client
        self.deployment_name = deployment_name
        self.id_field = id_field
        self.candidate_filter_field = candidate_filter_field
        self.vector_fields = vector_fields or ["chunk_content_vector"]
        self.bm25_search_fields = bm25_search_fields or ["chunk_content"]
        self.query_language = query_language
        self.select_fields = select_fields or [
            "file_name",
            "chunk_page_number",
            "chunk_function_summary",
            "chunk_content",
        ]
        self.use_semantic_reranking = use_semantic_reranking
        self.semantic_configuration_name = semantic_configuration_name
        self.max_candidate_filter_values = max_candidate_filter_values

    def _generate_embedding(self, text: str) -> List[float]:
        resp = self.openai_client.embeddings.create(input=text, model=self.deployment_name)
        return resp.data[0].embedding

    def _detect_language(self, q: str) -> str:
        # Basic detection: use zh-TW for Han characters, else en-US
        return "zh-TW" if re.search(r"[\u4e00-\u9fff]", q) else "en-US"

    def search(
        self,
        query: str,
        top: int = 10,
        stage1_candidates: int = 500,
        stage2_candidates: int = 50,
        *,
        filter_expression: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute hierarchical hybrid search.

        Args:
            query: Search query string
            top: Final number of results to return
            stage1_candidates: Candidate count from Stage 1 BM25
            stage2_candidates: Candidate count from Stage 2 Hybrid
            filter_expression: Optional OData filter applied in all stages
        """
        timings: Dict[str, float] = {}

        # Resolve language dynamically if not provided
        effective_language = self.query_language or self._detect_language(query)

        # ===== STAGE 1: Fast BM25 Pre-filtering =====
        start = time.perf_counter()
        stage1_results = list(
            self.search_client.search(
                search_text=query,
                search_fields=self.bm25_search_fields,
                query_language=effective_language,
                select=[self.id_field, self.candidate_filter_field],  # minimal payload + candidate grouping
                filter=filter_expression,
                top=stage1_candidates,
            )
        )
        # Extract candidate IDs (handle both dict and object cases)
        stage1_ids: List[str] = []
        stage1_candidate_values: List[str] = []
        for r in stage1_results:
            if isinstance(r, dict):
                rid = r.get(self.id_field)
                cval = r.get(self.candidate_filter_field)
            else:
                rid = getattr(r, self.id_field, None)
                cval = getattr(r, self.candidate_filter_field, None)
            if rid:
                stage1_ids.append(str(rid))
            if cval:
                stage1_candidate_values.append(str(cval))
        timings["stage1_ms"] = (time.perf_counter() - start) * 1000

        # If no IDs, bail out early
        if not stage1_ids:
            return []

        # ===== STAGE 2: Hybrid Search on Candidates =====
        start = time.perf_counter()
        query_vector = self._generate_embedding(query)
        vector_queries: List[VectorizedQuery] = [
            VectorizedQuery(vector=query_vector, k_nearest_neighbors=100, fields=vf)
            for vf in self.vector_fields
        ]

        # Restrict search within candidate values (prefer a filterable field) and apply optional external filters
        cf_filter = self._create_list_filter(self.candidate_filter_field, stage1_candidate_values)
        combined_filter = self._combine_filters(cf_filter, filter_expression)

        stage2_results = list(
            self.search_client.search(
                search_text=query,
                vector_queries=vector_queries,
                search_fields=self.bm25_search_fields,
                query_language=effective_language,
                filter=combined_filter,
                select=self.select_fields + [self.id_field],
                top=stage2_candidates,
            )
        )
        timings["stage2_ms"] = (time.perf_counter() - start) * 1000

        # ===== STAGE 3: Semantic Re-ranking (optional) =====
        start = time.perf_counter()
        if self.use_semantic_reranking and self.semantic_configuration_name:
            try:
                final_results = self._semantic_rerank(
                    query=query,
                    candidates=stage2_results,
                    top=top,
                    query_language=effective_language,
                )
            except Exception:
                # Fallback to top-K if semantic search is not available
                final_results = stage2_results[:top]
        else:
            final_results = stage2_results[:top]
        timings["stage3_ms"] = (time.perf_counter() - start) * 1000
        timings["total_ms"] = sum(timings.values())

        # Attach timing metadata
        for r in final_results:
            try:
                r["_timings"] = timings
            except TypeError:
                setattr(r, "_timings", timings)
        return final_results
    
    def _create_id_filter(self, ids: List[str]) -> Optional[str]:
        """Create OData filter for candidate ID list using search.in()."""
        if not ids:
            return None
        # For large ID lists, use search.in() function with comma separator
        id_list = ",".join(str(_id) for _id in ids[:1000])  # safety cap
        return f"search.in({self.id_field}, '{id_list}', ',')"

    def _combine_filters(self, a: Optional[str], b: Optional[str]) -> Optional[str]:
        if a and b:
            return f"({a}) and ({b})"
        return a or b

    def _create_list_filter(self, field_name: str, values: List[str]) -> Optional[str]:
        """Create search.in filter for a list of values on a given field using a safe separator."""
        if not values:
            return None
        # Use '|' as separator to avoid collisions with commas in filenames
        safe_sep = "|"
        # Preserve order while making values unique, then cap length
        seen = set()
        ordered_uniq: List[str] = []
        for v in values:
            # Escape single quotes per OData by doubling them
            sv = str(v).replace("'", "''")
            if sv not in seen:
                seen.add(sv)
                ordered_uniq.append(sv)
                if len(ordered_uniq) >= self.max_candidate_filter_values:
                    break
        uniq_values = ordered_uniq
        value_list = safe_sep.join(uniq_values)
        return f"search.in({field_name}, '{value_list}', '{safe_sep}')"
    
    def _semantic_rerank(
        self,
        query: str,
        candidates: List[Any],
        top: int,
        *,
        query_language: Optional[str] = None,
    ) -> List[Any]:
        """Re-rank candidates using Azure Cognitive Search semantic ranking."""
        # Extract candidate IDs robustly
        candidate_ids: List[str] = []
        for r in candidates:
            if isinstance(r, dict):
                rid = r.get(self.id_field)
            else:
                rid = getattr(r, self.id_field, None)
            if rid:
                candidate_ids.append(str(rid))
        filter_expr = self._create_id_filter(candidate_ids)

        semantic_results = self.search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name=self.semantic_configuration_name,
            query_language=query_language,
            filter=filter_expr,
            select=self.select_fields + [self.id_field],
            top=top,
        )
        return list(semantic_results)

# The module exposes HierarchicalHybridSearcher for use by the CLI or other callers.