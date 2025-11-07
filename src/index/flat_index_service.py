from __future__ import annotations

import logging

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    VectorSearch,
    VectorSearchProfile,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
)

logger = logging.getLogger(__name__)


class FlatIndexService:
    """
    Service to create a flat Azure AI Search index using the SDK.

    The index schema aligns with documents produced by
    `src/index/transformers/etl_flatten.py`.
    """

    def __init__(self, endpoint: str, api_key: str, index_name: str, openai_endpoint: str | None = None, openai_key: str | None = None, openai_deployment: str | None = None, embedding_dimensions: int = 1536):
        self.endpoint = endpoint
        self.api_key = api_key
        self.index_name = index_name
        self.embedding_dimensions = embedding_dimensions
        self.openai_endpoint = openai_endpoint
        self.openai_key = openai_key
        self.openai_deployment = openai_deployment
        self.index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))

    def create_index(self, recreate: bool = False) -> bool:
        try:
            existing = [idx.name for idx in self.index_client.list_indexes()]
            if self.index_name in existing:
                if recreate:
                    logger.warning(f"Deleting existing flat index: {self.index_name}")
                    self.index_client.delete_index(self.index_name)
                else:
                    logger.info(f"Flat index already exists: {self.index_name}")
                    return True

            # Define fields matching flattened docs
            fields = [
                SearchField(name="doc_id", type=SearchFieldDataType.String, key=True, filterable=True),

                # System/file metadata
                SearchField(name="sys_file_name", type=SearchFieldDataType.String, searchable=True, filterable=True),
                SearchField(name="sys_file_path", type=SearchFieldDataType.String, retrievable=True),
                SearchField(name="sys_file_size_bytes", type=SearchFieldDataType.Int64, filterable=True),
                SearchField(name="sys_file_type", type=SearchFieldDataType.String, filterable=True),
                SearchField(name="sys_last_updated", type=SearchFieldDataType.DateTimeOffset, filterable=True),
                SearchField(name="sys_page_count", type=SearchFieldDataType.Int32, filterable=True),
                SearchField(name="sys_extracted_at", type=SearchFieldDataType.DateTimeOffset, filterable=True),
                SearchField(name="sys_processing_version", type=SearchFieldDataType.String, filterable=True),

                # Indexing metadata
                SearchField(name="file_name", type=SearchFieldDataType.String, searchable=True, filterable=True),
                SearchField(name="library_name_en", type=SearchFieldDataType.String, searchable=True, filterable=True),
                SearchField(name="library_name_tc", type=SearchFieldDataType.String, searchable=True, filterable=True),
                SearchField(name="category_name_en", type=SearchFieldDataType.String, searchable=True, filterable=True),
                SearchField(name="category_name_tc", type=SearchFieldDataType.String, searchable=True, filterable=True),
                SearchField(name="title_name_en", type=SearchFieldDataType.String, searchable=True),
                SearchField(name="title_name_tc", type=SearchFieldDataType.String, searchable=True),
                SearchField(name="file_url", type=SearchFieldDataType.String, retrievable=True),
                SearchField(name="branch_name", type=SearchFieldDataType.String, filterable=True),
                SearchField(name="item_type", type=SearchFieldDataType.String, filterable=True),
                SearchField(name="item_url", type=SearchFieldDataType.String, retrievable=True),

                # Chunk content and vector
                SearchField(name="chunk_content", type=SearchFieldDataType.String, searchable=True),
                SearchField(
                    name="chunk_content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=self.embedding_dimensions,
                    vector_search_profile_name="flatHnswProfile",
                ),
                SearchField(name="chunk_page_number", type=SearchFieldDataType.Int32, filterable=True),
                SearchField(name="chunk_function_summary", type=SearchFieldDataType.String, searchable=True),

                # Chunk analytics and QA
                SearchField(name="chunk_char_count", type=SearchFieldDataType.Int32, filterable=True),
                SearchField(name="chunk_word_count", type=SearchFieldDataType.Int32, filterable=True),
                SearchField(name="chunk_sentence_count", type=SearchFieldDataType.Int32, filterable=True),
                SearchField(name="chunk_entities", type=SearchFieldDataType.Collection(SearchFieldDataType.String), searchable=True),
                SearchField(name="qa_questions", type=SearchFieldDataType.Collection(SearchFieldDataType.String), searchable=True),
                SearchField(name="qa_answers", type=SearchFieldDataType.Collection(SearchFieldDataType.String), searchable=True),
                SearchField(name="qa_confidence", type=SearchFieldDataType.Double, filterable=True),
            ]

            # Configure vector search (use OpenAI vectorizer if available)
            algorithms = [
                HnswAlgorithmConfiguration(
                    name="flatHnsw",
                    parameters=HnswParameters(metric="cosine", m=4, ef_construction=400, ef_search=500),
                )
            ]

            use_vectorizer = bool(self.openai_endpoint and self.openai_key and self.openai_deployment)
            profiles = [
                VectorSearchProfile(
                    name="flatHnswProfile",
                    algorithm_configuration_name="flatHnsw",
                    vectorizer_name="flatOpenAI" if use_vectorizer else None,
                )
            ]

            vectorizers = []
            if use_vectorizer:
                vectorizers = [
                    AzureOpenAIVectorizer(
                        vectorizer_name="flatOpenAI",
                        parameters=AzureOpenAIVectorizerParameters(
                            resource_url=self.openai_endpoint,
                            deployment_name=self.openai_deployment,
                            api_key=self.openai_key,
                        ),
                    )
                ]

            vector_search = VectorSearch(algorithms=algorithms, profiles=profiles, vectorizers=vectorizers)

            index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
            self.index_client.create_index(index)
            logger.info(f"Created flat index: {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create flat index: {str(e)}")
            return False