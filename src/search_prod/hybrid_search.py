import re
try:
    from azure.search.documents.models import QueryType, VectorizedQuery
except Exception:
    # Defer import issues to runtime environments that provide these
    QueryType = None  # type: ignore
    VectorizedQuery = None  # type: ignore


class AzureHybridSearch:
    """
    Production-ready hybrid search combining BM25 + vector search + RRF fusion.
    
    How RRF (Reciprocal Rank Fusion) works:
    1. BM25 search produces ranked list: [doc3, doc7, doc1, doc5, ...]
    2. Vector search produces ranked list: [doc1, doc3, doc9, doc2, ...]
    3. RRF assigns score to each doc: 1/(k + rank) where k=60 (default)
       - doc1: BM25 rank=3 → 1/63 = 0.016, Vector rank=1 → 1/61 = 0.016
         Combined: 0.016 + 0.016 = 0.032
       - doc3: BM25 rank=1 → 1/61 = 0.016, Vector rank=2 → 1/62 = 0.016
         Combined: 0.016 + 0.016 = 0.032
    4. Docs ranked by combined RRF score (higher = better)
    5. Top-k results returned to user
    
    Why RRF works: Balances both signals without manual weight tuning.
    Documents appearing high in BOTH rankings naturally score highest.
    """
    
    def __init__(self, search_client, openai_client):
        import os
        self.search_client = search_client
        self.openai_client = openai_client
        self.embedding_deployment = "text-embedding-3-small"
        self.vector_field = os.getenv("VECTOR_FIELD") or "content_chunk_vector"
        def _csv(name, default):
            v = os.getenv(name)
            return [x.strip() for x in v.split(",")] if v else default
        self.search_fields_en = _csv("SEARCH_FIELDS_EN", ["content_en", "title_name_en"])
        self.search_fields_tc = _csv("SEARCH_FIELDS_TC", ["content_tc", "title_name_tc"])
        self.select_fields = _csv(
            "SELECT_FIELDS",
            [
                "document_id",
                "doc_id",
                "title_name_en",
                "title_name_tc",
                "file_name",
                "page_number",
                "chunk_content",
                "content_en",
                "content_tc",
            ],
        )
    
    def create_hybrid_index(self):
        """
        Create hybrid search index supporting both text and vector.
        
        Key configuration choices:
        - content field: Searchable with analyzer for BM25
        - content_vector field: 1536-d vector for semantic search
        - Both can be searched simultaneously
        - m=8 for HNSW: Higher than pure vector (m=4) because hybrid
          benefits from better recall since BM25 also contributes
        """
        
        hybrid_index_config = {
            "name": "experiment_full_2",
            "fields": [
                {
                    "name": "id",
                    "type": "Edm.String",
                    "key": True
                },
                {"name": "content_en", "type": "Edm.String", "searchable": True, "analyzer": "en.microsoft"},
                {"name": "content_tc", "type": "Edm.String", "searchable": True, "analyzer": "zh-Hant.lucene"},
                {"name": "title_name_en", "type": "Edm.String", "searchable": True, "analyzer": "en.microsoft"},
                {"name": "title_name_tc", "type": "Edm.String", "searchable": True, "analyzer": "zh-Hant.lucene"},
                {"name": "content_chunk_vector", "type": "Collection(Edm.Single)", "searchable": True, "vectorSearchDimensions": 1536, "vectorSearchProfileName": "hybrid-vector-profile"},
                {"name": "filename", "type": "Edm.String", "filterable": True, "facetable": False, "searchable": False},
                {"name": "branch_name", "type": "Edm.String", "filterable": True, "facetable": True, "searchable": False},
                {"name": "document_id", "type": "Edm.String", "filterable": True, "facetable": True, "searchable": False},
                {"name": "entities", "type": "Collection(Edm.String)", "filterable": True, "facetable": True, "searchable": False},
                {"name": "page_number", "type": "Edm.Int32", "filterable": True, "sortable": True, "facetable": False},
                {"name": "word_count", "type": "Edm.Int32", "filterable": True, "sortable": True, "facetable": False},
                {"name": "char_count", "type": "Edm.Int32", "filterable": True, "sortable": True, "facetable": False},
                {"name": "lang_tags", "type": "Collection(Edm.String)", "filterable": True, "facetable": True, "searchable": False}
            ],
            "vectorSearch": {
                "algorithms": [
                    {
                        "name": "hybrid-hnsw",
                        "kind": "hnsw",
                        "hnswParameters": {
                            "metric": "cosine",
                            "m": 8,  # Higher for hybrid (vs. m=4 for pure vector)
                            "efConstruction": 400,
                            "efSearch": 500
                        }
                    }
                ],
                "profiles": [
                    {"name": "hybrid-vector-profile", "algorithm": "hybrid-hnsw"}
                ]
            }
        }
        
        return hybrid_index_config
    
    def hybrid_search(self, query, top_k=10, filter_expression=None, query_lang: str | None = None):
        """
        Perform hybrid search combining text and vector with RRF fusion.
        
        Returns standardized JSON payload:
        - query
        - search_method
        - total_result
        - results: unified fields across search modes
        """
        
        # Generate query embedding for vector component
        query_vector = self.embed_text(query)
        
        # Create vector query
        if VectorizedQuery is None:
            raise RuntimeError("VectorizedQuery model not available in this environment")
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k * 2,
            fields=self.vector_field
        )

        # Detect query language to route BM25 fields
        if query_lang is None:
            if re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]", query):
                search_fields = self.search_fields_tc
            elif re.search(r"[A-Za-z]", query):
                search_fields = self.search_fields_en
            else:
                search_fields = self.search_fields_en + self.search_fields_tc
        else:
            search_fields = self.search_fields_en if query_lang == "en" else self.search_fields_tc
        
        # Perform hybrid search (Azure AI Search handles RRF fusion automatically)
        if QueryType is None:
            raise RuntimeError("QueryType model not available in this environment")
        select_fields = [
            "document_id",
            "filename",
            "page_number",
            "branch_name",
            "title_name_en",
            "title_name_tc",
            "content_en",
            "content_tc",
            "lang_tags",
            "entities",
            "word_count",
            "char_count",
        ]
        results = self.search_client.search(
            search_text=query,
            search_fields=search_fields,
            vector_queries=[vector_query],
            filter=filter_expression,
            top=top_k,
            query_type=QueryType.SIMPLE,
            select=select_fields,
        )
        try:
            _ = next(iter(results))
            results = self.search_client.search(
                search_text=query,
                search_fields=search_fields,
                vector_queries=[vector_query],
                filter=filter_expression,
                top=top_k,
                query_type=QueryType.SIMPLE,
                select=select_fields,
            )
        except Exception:
            alt_field = "chunk_content_vector" if self.vector_field == "content_chunk_vector" else "content_chunk_vector"
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k * 2,
                fields=alt_field
            )
            results = self.search_client.search(
                search_text=query,
                search_fields=search_fields,
                vector_queries=[vector_query],
                filter=filter_expression,
                top=top_k,
                query_type=QueryType.SIMPLE,
            )
        
        items = []
        for r in results:
            rid = r.get("id")
            base_id = r.get("document_id") or (rid.split("_")[0] if isinstance(rid, str) and "_" in rid else rid)
            chunk = r.get("content_en") if (query_lang == "en") else (r.get("content_tc") if (query_lang == "tc") else (r.get("content_en") or r.get("content_tc")))
            items.append({
                "id": base_id,
                "document_id": base_id,
                "title_name_en": r.get("title_name_en") or "",
                "title_name_tc": r.get("title_name_tc") or "",
                "content_en": r.get("content_en") or "",
                "content_tc": r.get("content_tc") or "",
                "filename": r.get("filename") or r.get("file_name"),
                "page_number": r.get("page_number") or r.get("chunk_page_number"),
                "score": r.get("@search.score"),
                "content_chunk": chunk or "",
            })
        
        payload = {
            "query": query,
            "search_method": "hybrid",
            "total_result": len(items),
            "results": items,
        }
        return payload
    
    def hybrid_search_with_score_breakdown(self, query, top_k=10, query_lang: str | None = None):
        """
        Hybrid search with component score visibility (for debugging).
        
        Useful for understanding which component (BM25 vs. vector) is
        contributing more to final rankings. Helps diagnose issues like:
        - All results driven by BM25 → vector not adding value
        - All results driven by vector → BM25 not contributing
        - Mixed contributions → hybrid working as intended
        """
        
        query_vector = self.embed_text(query)
        
        # Get BM25-only results
        # BM25-only with language-aware fields
        if query_lang is None:
            if re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]", query):
                bm25_fields = self.search_fields_tc
            elif re.search(r"[A-Za-z]", query):
                bm25_fields = self.search_fields_en
            else:
                bm25_fields = self.search_fields_en + self.search_fields_tc
        else:
            bm25_fields = self.search_fields_en if query_lang == "en" else self.search_fields_tc

        bm25_results = self.search_client.search(
            search_text=query,
            search_fields=bm25_fields,
            top=top_k * 2
        )
        bm25_scores = {r["id"]: r["@search.score"] for r in bm25_results}
        
        # Get vector-only results
        if VectorizedQuery is None:
            raise RuntimeError("VectorizedQuery model not available in this environment")
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k * 2,
            fields=self.vector_field
        )
        vector_results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=top_k * 2
        )
        vector_scores = {r["id"]: r["@search.score"] for r in vector_results}
        
        # Get hybrid results
        if QueryType is None:
            raise RuntimeError("QueryType model not available in this environment")
        hybrid_results = self.search_client.search(
            search_text=query,
            search_fields=bm25_fields,
            vector_queries=[vector_query],
            top=top_k,
            query_type=QueryType.SIMPLE
        )
        
        # Annotate with component scores
        annotated_results = []
        for result in hybrid_results:
            chunk_id = result["id"]
            doc_display_id = result.get("document_id") or chunk_id
            annotated_results.append({
                "id": doc_display_id,
                "title_en": result.get("title_name_en", ""),
                "title_tc": result.get("title_name_tc", ""),
                "content_en": result.get("content_en", ""),
                "content_tc": result.get("content_tc", ""),
                "filename": result.get("filename"),
                "page_number": result.get("page_number"),
                "hybrid_score": result["@search.score"],
                "document_id": result.get("document_id"),
                "bm25_score": bm25_scores.get(chunk_id, 0.0),
                "vector_score": vector_scores.get(chunk_id, 0.0),
                "dominant_component": "BM25" if bm25_scores.get(chunk_id, 0) > vector_scores.get(chunk_id, 0) else "Vector"
            })
        
        return annotated_results

    def embed_text(self, text):
        """Generate embedding using Azure OpenAI."""
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_deployment
        )
        return response.data[0].embedding

    def hybrid_search_raw(self, query, top_k=10, filter_expression=None, query_lang: str | None = None):
        """
        Perform hybrid search and return raw Azure Search documents.

        This is useful for inspecting how data is indexed and verifying field mappings.
        """
        # Generate query embedding
        query_vector = self.embed_text(query)

        if VectorizedQuery is None:
            raise RuntimeError("VectorizedQuery model not available in this environment")

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k * 2,
            fields=self.vector_field,
        )

        # Language-aware BM25 routing
        if query_lang is None:
            if re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]", query):
                search_fields = self.search_fields_tc
            elif re.search(r"[A-Za-z]", query):
                search_fields = self.search_fields_en
            else:
                search_fields = self.search_fields_en + self.search_fields_tc
        else:
            search_fields = self.search_fields_en if query_lang == "en" else self.search_fields_tc

        if QueryType is None:
            raise RuntimeError("QueryType model not available in this environment")

        results = self.search_client.search(
            search_text=query,
            search_fields=search_fields,
            vector_queries=[vector_query],
            filter=filter_expression,
            top=top_k,
            query_type=QueryType.SIMPLE,
        )

        # Convert to plain dicts for JSON serialization
        raw_docs = []
        for r in results:
            try:
                raw_docs.append(dict(r))
            except Exception:
                # Fallback: manually gather known fields and search metadata
                raw_docs.append({
                    "id": r.get("id"),
                    "title_name_en": r.get("title_name_en"),
                    "title_name_tc": r.get("title_name_tc"),
                    "content_en": r.get("content_en"),
                    "content_tc": r.get("content_tc"),
                    "filename": r.get("filename"),
                    "page_number": r.get("page_number"),
                    "document_id": r.get("document_id"),
                    "branch_name": r.get("branch_name"),
                    "@search.score": r.get("@search.score"),
                    "@search.reranker_score": r.get("@search.reranker_score"),
                })
        return raw_docs