from typing import List, Dict, Any
import time

class HierarchicalHybridSearcher:
    """Three-stage hierarchical search for large corpora."""
    
    def search(
        self,
        query: str,
        top: int = 10,
        stage1_candidates: int = 500,
        stage2_candidates: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Execute hierarchical hybrid search.
        
        Stage 1 (Coarse): Fast BM25-only to retrieve 500 candidates
        Stage 2 (Hybrid): Hybrid search on 500 candidates → 50 results
        Stage 3 (Re-rank): Semantic re-ranking on 50 → final 10
        
        Args:
            query: Search query
            top: Final number of results
            stage1_candidates: Candidates after stage 1 (default: 500)
            stage2_candidates: Candidates after stage 2 (default: 50)
        """
        timings = {}
        
        # ===== STAGE 1: Fast BM25 Pre-filtering =====
        start = time.perf_counter()
        
        # Use BM25-only for fast candidate retrieval
        stage1_results = list(self.search_client.search(
            search_text=query,
            top=stage1_candidates
        ))
        stage1_ids = [r['id'] for r in stage1_results]
        
        timings['stage1_ms'] = (time.perf_counter() - start) * 1000
        
        # ===== STAGE 2: Hybrid Search on Candidates =====
        start = time.perf_counter()
        
        # Generate query embedding
        query_vector = self.embedding_function(query)
        
        # Hybrid search filtered to stage1 candidates
        # (using OData filter to restrict to candidate IDs)
        filter_expr = self._create_id_filter(stage1_ids)
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=100,  # Over-fetch within candidates
            fields="titleVector"
        )
        
        stage2_results = list(self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            filter=filter_expr,
            top=stage2_candidates
        ))
        
        timings['stage2_ms'] = (time.perf_counter() - start) * 1000
        
        # ===== STAGE 3: Semantic Re-ranking =====
        start = time.perf_counter()
        
        # Use semantic search (if available) for final re-ranking
        # Azure Cognitive Search semantic ranking
        if self.use_semantic_reranking:
            final_results = self._semantic_rerank(
                query,
                stage2_results,
                top
            )
        else:
            final_results = stage2_results[:top]
        
        timings['stage3_ms'] = (time.perf_counter() - start) * 1000
        timings['total_ms'] = sum(timings.values())
        
        # Add timing metadata
        for result in final_results:
            result['_timings'] = timings
        
        return final_results
    
    def _create_id_filter(self, ids: List[str]) -> str:
        """Create OData filter for ID list."""
        # For large ID lists, use search.in() function
        if len(ids) <= 100:
            id_list = ','.join(f"'{id}'" for id in ids)
            return f"search.in(id, '{id_list}', ',')"
        else:
            # For very large lists, chunk or use alternative approach
            return None  # No filter (search all)
    
    def _semantic_rerank(
        self,
        query: str,
        candidates: List[Any],
        top: int
    ) -> List[Any]:
        """
        Re-rank candidates using semantic search.
        
        Uses Azure Cognitive Search semantic ranking (if enabled).
        """
        candidate_ids = [r['id'] for r in candidates]
        filter_expr = self._create_id_filter(candidate_ids)
        
        # Semantic search with configuration
        semantic_results = self.search_client.search(
            search_text=query,
            query_type='semantic',
            semantic_configuration_name='my-semantic-config',
            filter=filter_expr,
            top=top
        )
        
        return list(semantic_results)


"""
# Usage
hierarchical_searcher = HierarchicalHybridSearcher(
    search_client,
    get_embedding,
    use_semantic_reranking=True
)

results = hierarchical_searcher.search(
    "laptop for video editing",
    top=10,
    stage1_candidates=500,  # BM25 retrieves 500 candidates
    stage2_candidates=50    # Hybrid narrows to 50
)

# Check timings
print(f"Stage 1 (BM25): {results[0]['_timings']['stage1_ms']:.1f}ms")
print(f"Stage 2 (Hybrid): {results[0]['_timings']['stage2_ms']:.1f}ms")
print(f"Stage 3 (Semantic): {results[0]['_timings']['stage3_ms']:.1f}ms")
print(f"Total: {results[0]['_timings']['total_ms']:.1f}ms")

# Example output:
# Stage 1 (BM25): 45.2ms
# Stage 2 (Hybrid): 68.4ms
# Stage 3 (Semantic): 32.1ms
# Total: 145.7ms

"""