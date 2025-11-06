"""
Azure AI Search service for document indexing.

Handles:
- Index creation with vector search configuration
- Document upload (single and batch)
- Index management
"""

import logging
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    VectorSearch,
    VectorSearchProfile,
)

from src.config import AzureOpenAIConfig, AzureSearchConfig
from src.models.document_models import SearchDocument

logger = logging.getLogger(__name__)


class SearchService:
    """
    Service for Azure AI Search operations.

    Handles index creation, document upload, and search queries.
    """

    def __init__(self, search_config: AzureSearchConfig, openai_config: AzureOpenAIConfig):
        """
        Initialize Search service.

        Args:
            search_config: Azure AI Search configuration
            openai_config: Azure OpenAI configuration (for vectorizer)
        """
        self.search_config = search_config
        self.openai_config = openai_config

        credential = AzureKeyCredential(search_config.key)

        # Initialize clients
        self.index_client = SearchIndexClient(
            endpoint=search_config.endpoint, credential=credential
        )

        self.search_client = SearchClient(
            endpoint=search_config.endpoint,
            index_name=search_config.index_name,
            credential=credential,
        )

        logger.info(
            f"Initialized SearchService: {search_config.endpoint}/{search_config.index_name}"
        )

    def create_index(self, recreate: bool = False) -> bool:
        """
        Create Azure AI Search index with vector search configuration.

        Creates index with exact schema:
        - id, filename, file_summary, file_summary_chunk (vector 1536),
        - metadata_file (JSON), content_chunk, metadata_chunk (JSON),
        - content_chunk_dim (vector 1536)

        Args:
            recreate: If True, delete and recreate index if it exists

        Returns:
            bool: True if index created/exists, False otherwise

        Example:
            >>> search_service = SearchService(search_config, openai_config)
            >>> search_service.create_index()
        """
        index_name = self.search_config.index_name

        try:
            # Check if index exists
            existing_indexes = [idx.name for idx in self.index_client.list_indexes()]

            if index_name in existing_indexes:
                if recreate:
                    logger.warning(f"Deleting existing index: {index_name}")
                    self.index_client.delete_index(index_name)
                else:
                    logger.info(f"Index already exists: {index_name}")
                    return True

            logger.info(f"Creating index: {index_name}")

            # Define fields (exact schema from requirements)
            fields = [
                SearchField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True,
                ),
                SearchField(
                    name="filename",
                    type=SearchFieldDataType.String,
                    searchable=True,
                    retrievable=True,
                    filterable=True,
                ),
                SearchField(
                    name="file_summary",
                    type=SearchFieldDataType.String,
                    searchable=True,
                    retrievable=True,
                ),
                SearchField(
                    name="file_summary_chunk",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="myHnswProfile",
                ),
                SearchField(
                    name="metadata_file",
                    type=SearchFieldDataType.String,
                    retrievable=True,
                    filterable=True,
                ),
                SearchField(
                    name="content_chunk",
                    type=SearchFieldDataType.String,
                    searchable=True,
                    retrievable=True,
                ),
                SearchField(
                    name="metadata_chunk",
                    type=SearchFieldDataType.String,
                    retrievable=True,
                    filterable=True,
                ),
                SearchField(
                    name="content_chunk_dim",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="myHnswProfile",
                ),
            ]

            # Configure vector search with HNSW algorithm
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="myHnsw",
                        parameters=HnswParameters(
                            metric="cosine",
                            m=4,
                            ef_construction=400,
                            ef_search=500,
                        ),
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="myHnswProfile",
                        algorithm_configuration_name="myHnsw",
                        vectorizer_name="myOpenAI",
                    )
                ],
                vectorizers=[
                    AzureOpenAIVectorizer(
                        vectorizer_name="myOpenAI",
                        parameters=AzureOpenAIVectorizerParameters(
                            resource_url=self.openai_config.endpoint,
                            deployment_name=self.openai_config.deployment_name,
                            model_name="text-embedding-3-small",
                        ),
                    )
                ],
            )

            # Configure semantic search (optional but recommended)
            semantic_config = SemanticConfiguration(
                name="my-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="filename"),
                    content_fields=[
                        SemanticField(field_name="content_chunk"),
                        SemanticField(field_name="file_summary"),
                    ],
                ),
            )

            semantic_search = SemanticSearch(configurations=[semantic_config])

            # Create index
            index = SearchIndex(
                name=index_name,
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search,
            )

            self.index_client.create_index(index)
            logger.info(f"✅ Successfully created index: {index_name}")

            return True

        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {str(e)}")
            return False

    def upload_document(self, document: SearchDocument) -> bool:
        """
        Upload a single document to the index.

        Args:
            document: SearchDocument to upload

        Returns:
            bool: True if upload successful

        Example:
            >>> doc = SearchDocument(id="doc1_p1_c1", filename="doc1.pdf", ...)
            >>> search_service.upload_document(doc)
        """
        try:
            result = self.search_client.upload_documents(documents=[document.to_dict()])

            if result[0].succeeded:
                logger.debug(f"Uploaded document: {document.id}")
                return True
            else:
                logger.error(f"Failed to upload document {document.id}: {result[0].error_message}")
                return False

        except Exception as e:
            logger.error(f"Exception uploading document {document.id}: {str(e)}")
            return False

    def upload_documents_batch(
        self, documents: list[SearchDocument], batch_size: int = 1000
    ) -> dict[str, Any]:
        """
        Upload multiple documents in batches.

        Args:
            documents: List of SearchDocuments to upload
            batch_size: Number of documents per batch (max 1000)

        Returns:
            dict: Upload statistics

        Example:
            >>> docs = [SearchDocument(...), SearchDocument(...), ...]
            >>> stats = search_service.upload_documents_batch(docs)
            >>> print(f"Uploaded: {stats['successful']}/{stats['total']}")
        """
        if not documents:
            logger.warning("No documents to upload")
            return {"total": 0, "successful": 0, "failed": 0}

        logger.info(f"Uploading {len(documents)} documents in batches of {batch_size}")

        total = len(documents)
        successful = 0
        failed = 0

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(documents) - 1) // batch_size + 1

            logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")

            try:
                batch_dicts = [doc.to_dict() for doc in batch]
                results = self.search_client.upload_documents(documents=batch_dicts)

                batch_successful = sum(1 for r in results if r.succeeded)
                batch_failed = len(results) - batch_successful

                successful += batch_successful
                failed += batch_failed

                if batch_failed > 0:
                    logger.warning(f"Batch {batch_num}: {batch_failed} documents failed")

            except Exception as e:
                logger.error(f"Exception in batch {batch_num}: {str(e)}")
                failed += len(batch)

        logger.info(
            f"Upload complete: {successful}/{total} successful, {failed} failed "
            f"({(successful/total*100):.1f}% success rate)"
        )

        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0,
        }

    def index_exists(self) -> bool:
        """
        Check if the index exists.

        Returns:
            bool: True if index exists
        """
        try:
            existing_indexes = [idx.name for idx in self.index_client.list_indexes()]
            exists = self.search_config.index_name in existing_indexes
            logger.debug(
                f"Index {self.search_config.index_name} exists: {exists}"
            )
            return exists
        except Exception as e:
            logger.error(f"Error checking index existence: {str(e)}")
            return False


class NullSearchService:
    """
    No-op SearchService used in offline mode.

    Implements the same interface but does not perform any network operations.
    """

    def __init__(self):
        logger.info("Initialized NullSearchService (offline, no network calls)")

    def create_index(self, recreate: bool = False) -> bool:
        logger.info("[OFFLINE] create_index: no-op")
        return True

    def upload_document(self, document: SearchDocument) -> bool:
        logger.debug(f"[OFFLINE] upload_document: {getattr(document, 'id', 'unknown')}")
        return True

    def upload_documents_batch(self, documents: list[SearchDocument], batch_size: int = 1000) -> dict[str, Any]:
        logger.info(f"[OFFLINE] upload_documents_batch: {len(documents)} docs, no-op")
        return {"total": len(documents), "successful": len(documents), "failed": 0}

    def index_exists(self) -> bool:
        logger.info("[OFFLINE] index_exists: returning True")
        return True

    def get_document_count(self) -> int:
        logger.info("[OFFLINE] get_document_count: returning 0")
        return 0

    def close(self):
        logger.info("[OFFLINE] close: no-op")

    def get_document_count(self) -> int:
        """
        Get the number of documents in the index.

        Returns:
            int: Document count, -1 if error
        """
        try:
            # Search with empty query to get count
            results = self.search_client.search(
                search_text="*", include_total_count=True, top=0
            )
            count = results.get_count()
            logger.debug(f"Index contains {count} documents")
            return count if count is not None else 0
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            return -1

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document by ID.

        Args:
            document_id: ID of document to delete

        Returns:
            bool: True if deletion successful
        """
        try:
            result = self.search_client.delete_documents(documents=[{"id": document_id}])
            if result[0].succeeded:
                logger.info(f"Deleted document: {document_id}")
                return True
            else:
                logger.error(f"Failed to delete {document_id}: {result[0].error_message}")
                return False
        except Exception as e:
            logger.error(f"Exception deleting document {document_id}: {str(e)}")
            return False

    def delete_documents_by_filename(self, filename: str) -> int:
        """
        Delete all documents (chunks) for a specific filename.

        Useful for re-processing a document.

        Args:
            filename: Filename to delete

        Returns:
            int: Number of documents deleted
        """
        try:
            # Search for all documents with this filename
            results = self.search_client.search(
                search_text="*",
                filter=f"filename eq '{filename}'",
                select="id",
            )

            doc_ids = [doc["id"] for doc in results]

            if not doc_ids:
                logger.info(f"No documents found for filename: {filename}")
                return 0

            logger.info(f"Deleting {len(doc_ids)} documents for filename: {filename}")

            # Delete in batches
            deleted = 0
            for i in range(0, len(doc_ids), 1000):
                batch = doc_ids[i : i + 1000]
                batch_docs = [{"id": doc_id} for doc_id in batch]
                results = self.search_client.delete_documents(documents=batch_docs)
                deleted += sum(1 for r in results if r.succeeded)

            logger.info(f"Deleted {deleted}/{len(doc_ids)} documents for {filename}")
            return deleted

        except Exception as e:
            logger.error(f"Error deleting documents for {filename}: {str(e)}")
            return 0

    def close(self):
        """Close search clients."""
        if self.search_client:
            self.search_client.close()
        if self.index_client:
            self.index_client.close()
        logger.info("Closed SearchService")


# Example usage
if __name__ == "__main__":
    import sys

    from src.config import get_config

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        config = get_config()
        search_service = SearchService(config.azure_search, config.azure_openai)

        print(f"Azure AI Search Service")
        print(f"Endpoint: {config.azure_search.endpoint}")
        print(f"Index: {config.azure_search.index_name}\n")

        # Check if index exists
        if search_service.index_exists():
            print(f"✅ Index exists: {config.azure_search.index_name}")
            count = search_service.get_document_count()
            print(f"📊 Document count: {count}")
        else:
            print(f"❌ Index does not exist: {config.azure_search.index_name}")
            print("\nCreating index...")
            if search_service.create_index():
                print("✅ Index created successfully!")
            else:
                print("❌ Failed to create index")
                sys.exit(1)

        search_service.close()

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
