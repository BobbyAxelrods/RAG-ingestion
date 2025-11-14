from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
import numpy as np
import os
try:
    from azure.search.documents.models import VectorizedQuery
except Exception:
    VectorizedQuery = None  # type: ignore

class AzureVectorSearch:
    """
    Production-ready vector search using Azure OpenAI + Azure AI Search.
    
    Architecture:
    1. Documents embedded via Azure OpenAI (text-embedding-ada-002 or text-embedding-3-small)
    2. Vectors stored in Azure AI Search vector fields
    3. Query embedded with same model
    4. HNSW (Hierarchical Navigable Small World) approximate nearest neighbor search
    5. Results ranked by cosine similarity
    """
    
    def __init__(self, search_endpoint, search_key, openai_endpoint, openai_key, index_name):
        # Initialize Azure AI Search client
        self.search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(search_key)
        )
        
        # Initialize Azure OpenAI client
        self.openai_client = AzureOpenAI(
            azure_endpoint=openai_endpoint,
            api_key=openai_key,
            api_version="2024-02-01"
        )
        self.embedding_deployment = "text-embedding-3-small"  # Or "text-embedding-3-small"
        vf_env = os.getenv("VECTOR_FIELD")
        self.vector_field_candidates = [
            vf_env,
            "content_chunk_vector",
            "chunk_content_vector",
            "content_vector",
        ]
        self.vector_field_candidates = [v for v in self.vector_field_candidates if v]
    
    def get_embedding(self, text):
        """
        Generate embedding for text using Azure OpenAI.
        
        Args:
            text: Text to embed (max 8191 tokens for ada-002)
        
        Returns:
            1536-dimensional vector (ada-002) or 1536-d (text-embedding-3-small)
        
        Cost: $0.0001 per 1K tokens (~$0.10 per 1M tokens)
        """
        
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_deployment
        )
        
        return response.data[0].embedding
    
    def create_vector_index(self):
        """
        Create vector search index configuration.
        
        Key parameters:
        - vectorSearchDimensions: Must match embedding model (1536 for ada-002)
        - metric: "cosine" (most common), "euclidean", or "dotProduct"
        - HNSW parameters:
          * m: Number of bi-directional links (4-16, higher = better recall, slower build)
          * efConstruction: Candidates during build (100-1000, higher = better quality, slower build)
          * efSearch: Candidates during search (100-1000, higher = better recall, slower query)
        """
        
        vector_index_config = {
            "name": "vector-search-index",
            "fields": [
                {
                    "name": "id",
                    "type": "Edm.String",
                    "key": True,
                    "searchable": False
                },
                {
                    "name": "content",
                    "type": "Edm.String",
                    "searchable": True  # Still searchable for hybrid
                },
                {
                    "name": "content_vector",
                    "type": "Collection(Edm.Single)",  # Vector field
                    "searchable": True,
                    "vectorSearchDimensions": 1536,  # ada-002 dimension
                    "vectorSearchProfileName": "vector-profile"
                },
                {
                    "name": "title",
                    "type": "Edm.String",
                    "searchable": True
                },
                {
                    "name": "metadata",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": True
                }
            ],
            "vectorSearch": {
                "algorithms": [
                    {
                        "name": "hnsw-algorithm",
                        "kind": "hnsw",
                        "hnswParameters": {
                            "metric": "cosine",  # Cosine similarity
                            "m": 4,  # 4 is default, good balance
                            "efConstruction": 400,  # Build quality (400 is default)
                            "efSearch": 500  # Search quality (500 is default)
                        }
                    }
                ],
                "profiles": [
                    {
                        "name": "vector-profile",
                        "algorithm": "hnsw-algorithm"
                    }
                ]
            }
        }
        
        return vector_index_config
    
    def embed_text(self, text):
        """
        Generate embedding using Azure OpenAI.
        
        Batch optimization: For indexing operations, batch multiple texts
        to reduce API calls and improve throughput.
        
        Cost implications:
        - text-embedding-ada-002: $0.0001/1K tokens (~750 words)
        - text-embedding-3-small: $0.00002/1K tokens (5x cheaper)
        - text-embedding-3-large: $0.00013/1K tokens (30% more expensive, 3072 dimensions)
        
        Typical costs for 10,000 documents (avg 500 tokens each):
        - ada-002: $0.50 for initial indexing
        - 3-small: $0.10 for initial indexing
        - Query costs: $0.0001 per query (ada-002)
        
        Dimension tradeoffs:
        - 1536-d (ada-002, 3-small): Good balance, most common
        - 3072-d (3-large): Better accuracy (+5-10%), 2x storage cost, slower search
        - Smaller dimensions (512-d, 768-d): Possible with text-embedding-3-small/large
          but requires dimension parameter and loses some quality
        """
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_deployment
        )
        return response.data[0].embedding
    
    def embed_batch(self, texts, batch_size=16):
        """
        Batch embed multiple texts for efficiency.
        
        Azure OpenAI supports up to 16 texts per API call for embedding models.
        This reduces API overhead and improves throughput for indexing operations.
        
        Example indexing throughput:
        - Sequential (1 at a time): ~10 docs/sec
        - Batched (16 at a time): ~120 docs/sec (12x faster)
        - Parallel batches (4 concurrent): ~400 docs/sec
        
        Args:
            texts: List of strings to embed
            batch_size: Number of texts per API call (max 16)
        
        Returns:
            List of embeddings (same order as input texts)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.openai_client.embeddings.create(
                input=batch,
                model=self.embedding_deployment
            )
            
            # Extract embeddings in order
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def vector_search(self, query, top_k=10, filter_expression=None, query_lang: str | None = None):
        """
        Perform vector search with Azure AI Search.
        
        Returns standardized JSON payload:
        - query
        - search_method
        - total_result
        - results: list where each item includes unified fields
        """
        
        # Generate query embedding
        query_vector = self.embed_text(query)
        
        # Create vector query
        if VectorizedQuery is None:
            raise RuntimeError("VectorizedQuery model not available in this environment")
        # Try multiple vector fields to align with index schema
        
        # Search with vector
        desired_select = [
            "document_id",
            "doc_id",
            "file_name",
            "page_number",
            "chunk_page_number",
            "chunk_content",
            "content_en",
            "content_tc",
            "branch_name",
            "entities",
        ]
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
        last_error = None
        results = None
        for vf in self.vector_field_candidates:
            try:
                vector_query = VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=top_k,
                    fields=vf
                )
                results = self.search_client.search(
                    search_text=None,
                    vector_queries=[vector_query],
                    filter=filter_expression,
                    top=top_k,
                    select=select_fields,
                )
                _ = next(iter(results))
                results = self.search_client.search(
                    search_text=None,
                    vector_queries=[vector_query],
                    filter=filter_expression,
                    top=top_k,
                    select=select_fields,
                )
                break
            except Exception as e:
                last_error = e
                continue
        if results is None:
            raise last_error or RuntimeError("No valid vector field matched for index")
        
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
                "page_number": r.get("page_number") or r.get("chunk_page_number") or r.get("page"),
                "score": r.get("@search.score") or r.get("score"),
                "content_chunk": chunk or "",
            })
        
        payload = {
            "query": query,
            "search_method": "vector",
            "total_result": len(items),
            "results": items,
        }
        return payload

"""
**Understanding Vector Search Performance**

Vector search latency breakdown (typical P95 values):
- **Embedding generation**: 15-30ms (Azure OpenAI API call)
- **Vector search (HNSW)**: 10-40ms depending on index size and efSearch
- **Result retrieval**: 5-10ms (fetch document content from storage)
- **Total**: 30-80ms for P95 latency

Factors affecting vector search speed:
1. **Index size**: 
   - 10K docs: ~10ms search time
   - 100K docs: ~20ms search time
   - 1M docs: ~40ms search time
   - 10M docs: ~80ms search time (consider sharding)

2. **HNSW parameters**:
   - m=4: Faster search, lower recall
   - m=8: Balanced (recommended for most use cases)
   - m=16: Slower search, higher recall
   - efSearch=100: Fast but recall ~0.80
   - efSearch=500: Default, recall ~0.95
   - efSearch=1000: Slower, recall ~0.98

3. **Dimension count**:
   - 1536-d (ada-002): Standard, good performance
   - 3072-d (3-large): 30-50% slower due to larger vectors
   - 512-d: Faster but lower quality

Cost efficiency for vector search:

"""