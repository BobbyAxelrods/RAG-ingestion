from typing import List, Dict, Any, Optional, Union
from azure.search.documents.models import VectorizedQuery


class HybridSearcher:
    """
    Execute hybrid search (BM25 + vector) using Azure Search.

    - Accepts multiple vector fields for multi-vector queries.
    - Reuses an embedding client to generate the query vector.
    - Returns Azure Search documents as a list for direct consumption.
    """

    def __init__(self, search_client, openai_client, deployment_name: str = "text-embedding-small-3") -> None:
        self.search_client = search_client
        self.openai_client = openai_client
        self.deployment_name = deployment_name

    def generate_embedding(self, query_text: str) -> List[float]:
        resp = self.openai_client.embeddings.create(input=query_text, model=self.deployment_name)
        return resp.data[0].embedding

    def hybrid_search(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        vector_fields: List[str] = None,
        k_nearest: int = 50,
        filter_expression: Optional[str] = None,
        select_fields: Optional[List[str]] = None,
        search_fields: Optional[Union[str, List[str]]] = None,
        query_language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not vector_fields:
            vector_fields = ["chunk_content_vector"]

        embedding = self.generate_embedding(query_text)
        vector_queries: List[VectorizedQuery] = []
        for vf in vector_fields:
            vector_queries.append(
                VectorizedQuery(vector=embedding, k_nearest_neighbors=k_nearest, fields=vf)
            )

        # Default select fields if not provided
        effective_select = select_fields or ["file_name", "chunk_page_number", "chunk_function_summary", "chunk_content"]

        results = self.search_client.search(
            search_text=query_text,
            vector_queries=vector_queries,
            top=top_k,
            filter=filter_expression,
            select=effective_select,
            search_fields=search_fields,
            query_language=query_language,
        )
        return list(results)