from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import numpy as np

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
        self.embedding_deployment = "text-embedding-ada-002"  # Or "text-embedding-3-small"
    
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
    
    def vector_search(self, query, top_k=10, filter_expression=None):
        """
        Perform vector search with Azure AI Search.
        
        How HNSW search works:
        1. Query embedding generated via Azure OpenAI
        2. HNSW algorithm navigates graph structure to find approximate nearest neighbors
        3. efSearch parameter controls search quality vs speed:
           - 100: Fast but lower recall (~0.80)
           - 500: Default balance (recall ~0.95)
           - 1000: Slower but better recall (~0.98)
        4. Results ranked by cosine similarity (-1 to 1, higher = more similar)
        
        Performance tuning:
        - top_k=10: Typical for user-facing search (30-50ms)
        - top_k=100: For reranking pipelines (80-120ms)
        - Filters add 10-30ms depending on selectivity
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            filter_expression: OData filter (e.g., "category eq 'mortgage'")
        
        Returns:
            List of results with scores (cosine similarity 0-1)
        """
        
        # Generate query embedding
        query_vector = self.embed_text(query)
        
        # Create vector query
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="content_vector"
        )
        
        # Search with vector
        results = self.search_client.search(
            search_text=None,  # Pure vector search (no text component)
            vector_queries=[vector_query],
            filter=filter_expression,
            top=top_k
        )
        
        return [
            {
                "id": result["id"],
                "content": result["content"],
                "score": result["@search.score"],  # Cosine similarity
                "title": result.get("title", "")
            }
            for result in results
        ]

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