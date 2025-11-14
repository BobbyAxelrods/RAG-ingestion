class HybridSearchDiagnostics:
    """Diagnose hybrid search result composition."""
    
    def analyze_result_sources(
        self,
        query: str,
        top: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze which search mode contributes which results.
        
        Returns:
            Analysis showing BM25 vs vector contribution to final results
        """
        # Execute BM25-only search
        bm25_results = self.search_client.search(
            search_text=query,
            top=top
        )
        bm25_ids = {r['id']: rank for rank, r in enumerate(bm25_results, 1)}
        
        # Execute vector-only search
        query_vector = self.embedding_function(query)
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top,
            fields="titleVector"
        )
        vector_results = self.search_client.search(
            vector_queries=[vector_query],
            top=top
        )
        vector_ids = {r['id']: rank for rank, r in enumerate(vector_results, 1)}
        
        # Execute hybrid search
        hybrid_results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=top
        )
        hybrid_ids = [r['id'] for r in hybrid_results]
        
        # Analyze composition
        analysis = {
            'query': query,
            'bm25_count': len(bm25_ids),
            'vector_count': len(vector_ids),
            'hybrid_top10_sources': [],
            'bm25_in_top10': 0,
            'vector_in_top10': 0,
            'both_in_top10': 0
        }
        
        for rank, doc_id in enumerate(hybrid_ids[:10], 1):
            in_bm25 = doc_id in bm25_ids
            in_vector = doc_id in vector_ids
            
            source = 'both' if (in_bm25 and in_vector) else ('bm25' if in_bm25 else 'vector')
            
            analysis['hybrid_top10_sources'].append({
                'rank': rank,
                'id': doc_id,
                'source': source,
                'bm25_rank': bm25_ids.get(doc_id),
                'vector_rank': vector_ids.get(doc_id)
            })
            
            if source == 'both':
                analysis['both_in_top10'] += 1
            elif source == 'bm25':
                analysis['bm25_in_top10'] += 1
            else:
                analysis['vector_in_top10'] += 1
        
        # Calculate dominance score
        analysis['vector_dominance'] = analysis['vector_in_top10'] / 10.0  # 0-1 score
        analysis['bm25_dominance'] = analysis['bm25_in_top10'] / 10.0
        
        return analysis

"""
# Usage
diagnostics = HybridSearchDiagnostics(search_client, get_embedding)
analysis = diagnostics.analyze_result_sources("Dell XPS 9520", top=20)

print(f"Query: {analysis['query']}")
print(f"Vector dominance: {analysis['vector_dominance']:.1%}")  # 0.8 = 80% vector-only results
print(f"BM25 dominance: {analysis['bm25_dominance']:.1%}")      # 0.1 = 10% BM25-only results
print(f"Both sources: {analysis['both_in_top10']}")             # 1 = 10% from both

print("\nTop 10 composition:")
for item in analysis['hybrid_top10_sources']:
    print(f"  #{item['rank']}: {item['source']:6} (BM25 #{item['bm25_rank']}, Vector #{item['vector_rank']})")

# Example output showing vector dominance:
# Query: Dell XPS 9520
# Vector dominance: 80.0%
# BM25 dominance: 10.0%
# Both sources: 1
#
# Top 10 composition:
#   #1: vector (BM25 #None, Vector #1)
#   #2: vector (BM25 #None, Vector #2)
#   #3: vector (BM25 #None, Vector #3)
#   #4: both   (BM25 #1, Vector #5)     ← Exact match at #4, should be #1
#   #5: vector (BM25 #None, Vector #4)
#   ...

"""