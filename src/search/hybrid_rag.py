import re
from typing import List, Dict, Any, Optional

from azure.search.documents.models import VectorizedQuery


class HybridRAGSystem:
    """RAG system using hybrid search aligned to your index schema.

    - Uses `chunk_content_vector` for vector search (1536 dims, cosine).
    - Selects fields that exist in your index: `file_name`, `chunk_content`, `file_url`, `chunk_page_number`, etc.
    - Builds citations with file name and page number per your artifact JSON.
    """

    def __init__(self, search_client, embedding_function, openai_client):
        self.search_client = search_client
        self.embedding_function = embedding_function
        self.openai_client = openai_client

    def _is_traditional_chinese(self, text: str) -> bool:
        return bool(re.search(r"[\u4E00-\u9FFF]", text))

    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        k_neighbors: int = 50,
        model: str = "gpt-4o-mini",
        document_id_filter: Optional[str] = None,
        branch_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate answer using hybrid search (BM25 + vector) and build proper citations.

        Args:
            query: User question
            top_k: Number of chunks to retrieve
            k_neighbors: K for vector stage
            model: Azure OpenAI deployment name
            document_id_filter: Optional filter (e.g., "GI000001")
            branch_filter: Optional filter (e.g., "HK" or "MACAU")

        Returns:
            Dict with generated answer, sources, and metadata
        """
        # ===== Step 1: Hybrid Search Retrieval (aligned to schema) =====
        query_vector: List[float] = self.embedding_function(query)

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=k_neighbors,
            fields="chunk_content_vector",
        )

        # Optional filter composition using filterable fields in schema
        filters: List[str] = []
        if document_id_filter:
            filters.append(f"document_id eq '{document_id_filter}'")
        if branch_filter:
            filters.append(f"branch_name eq '{branch_filter}'")
        filter_expr = " and ".join(filters) if filters else None

        # Choose semantic configuration based on language
        semantic_config = (
            "insurance-semantic-config-tc" if self._is_traditional_chinese(query) else "insurance-semantic-config-en"
        )

        search_results = list(
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
                # Note: leave query_type default; semantic config still helps ranking in preview SDKs
            )
        )

        # ===== Step 2: Build Context from Results (citations) =====
        context_parts: List[str] = []
        sources: List[Dict[str, Any]] = []

        for rank, result in enumerate(search_results, 1):
            file_name = result.get("file_name")
            page = result.get("chunk_page_number")
            content = result.get("chunk_content")
            url = result.get("file_url")
            doc_id = result.get("doc_id")

            # Compose context with clear source markers
            context_parts.append(
                f"[Source {rank}] {file_name} (page {page})\n{content}\n"
            )

            sources.append(
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "page": page,
                    "url": url,
                    "document_id": result.get("document_id"),
                    "library_name_en": result.get("library_name_en"),
                    "library_name_tc": result.get("library_name_tc"),
                    "category_name_en": result.get("category_name_en"),
                    "category_name_tc": result.get("category_name_tc"),
                    "search_score": result.get("@search.score"),
                }
            )

        context = "\n".join(context_parts)

        # ===== Step 3: Generate Answer with Azure OpenAI =====
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
            temperature=0.2,
            max_tokens=700,
        )

        generated_answer = response.choices[0].message.content

        return {
            "query": query,
            "answer": generated_answer,
            "sources": sources,
            "retrieval_method": "hybrid_search",
            "model": model,
            "token_usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }


# Example usage (configure clients externally in your app):
# from openai import AzureOpenAI
# openai_client = AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version="2024-02-15-preview",
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
# )
# search_client = ...  # azure.search.documents.SearchClient
# embedding_function = ...  # returns 1536-dim list for 'chunk_content_vector'
# rag_system = HybridRAGSystem(search_client, embedding_function, openai_client)
# result = rag_system.generate_answer(
#     query="澳門 CRS Form 的提交要求是什麼？",
#     top_k=6,
#     k_neighbors=100,
#     document_id_filter="GI000001",  # optional, targets GI document
# )
# print("Question:", result["query"]) 
# print("\nAnswer:", result["answer"]) 
# print("\nSources:")
# for source in result["sources"]:
#     print(f"  [{{source['rank']}}] {{source['file_name']}} (page {{source['page']}})")
# print(f"\nTokens used: {{result['token_usage']['total_tokens']}}")