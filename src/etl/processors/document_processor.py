"""
Document Processor Orchestrator.

Orchestrates the complete document processing pipeline:
1. Download from Blob Storage
2. Document Intelligence extraction
3. Image processing (OCR + descriptions)
4. Metadata enrichment (CSV lookup)
5. File summary generation
6. Chunking
7. Embedding generation
8. Upload to Azure AI Search

Handles both DocAnalysis (PDFs, Office, images) and LangChain (text files) paths.
"""

import logging
from pathlib import Path
from typing import Any

from src.etl.chunker.doc_analysis_chunker import DocAnalysisChunker
from src.etl.chunker.langchain_chunker import LangChainChunker
from src.etl.config import Config
from src.etl.models.document_models import DocumentType, EnrichedDocument, SearchDocument
from src.etl.services.blob_service import BlobStorageService
from src.etl.services.doc_intel_service import DocumentIntelligenceService
from src.etl.services.image_processing_service import ImageProcessingService
from src.etl.services.metadata_enrichment_service import MetadataEnrichmentService
from src.etl.services.openai_service import OpenAIService
from src.etl.services.search_service import SearchService, NullSearchService
from src.etl.services.summary_generation_service import SummaryGenerationService

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Orchestrates end-to-end document processing pipeline.

    Supports two processing paths:
    - DocAnalysis path: PDFs, DOCX, XLSX, PPTX, images
    - LangChain path: TXT, MD, CSV, JSON

    Features:
    - Automatic file type detection
    - Error handling with graceful degradation
    - Progress tracking
    - Batch processing support
    """

    def __init__(self, config: Config, enable_search: bool = True):
        """
        Initialize Document Processor.

        Args:
            config: Complete configuration object
        """
        self.config = config

        # Initialize all services
        logger.info("Initializing DocumentProcessor services...")

        self.blob_service = BlobStorageService(config.azure_blob)
        self.doc_intel_service = DocumentIntelligenceService(config.doc_intelligence)
        self.openai_service = OpenAIService(config.azure_openai, config.azure_openai_vision)
        self.image_service = ImageProcessingService(
            config.image_processing, self.doc_intel_service, self.openai_service
        )
        self.metadata_service = MetadataEnrichmentService(config.metadata)
        self.summary_service = SummaryGenerationService(config.file_summary)
        # Initialize search service only when enabled; otherwise use no-op service
        self.search_service = (
            SearchService(config.azure_search, config.azure_openai)
            if enable_search
            else NullSearchService()
        )

        # Initialize chunkers
        self.doc_analysis_chunker = DocAnalysisChunker(config.chunking)
        self.langchain_chunker = LangChainChunker(config.chunking)

        logger.info("DocumentProcessor initialized successfully")

    def process_document(
        self,
        filename: str,
        file_bytes: bytes | None = None,
        skip_index: bool = False,
    ) -> dict[str, Any]:
        """
        Process a single document through the complete pipeline.

        Args:
            filename: Document filename
            file_bytes: Optional file bytes (if not provided, downloads from Blob)

        Returns:
            dict: Processing results with statistics

        Example:
            >>> processor = DocumentProcessor(config)
            >>> results = processor.process_document("policy.pdf")
            >>> print(f"Uploaded {results['chunks_uploaded']} chunks")
        """
        logger.info(f"=" * 80)
        logger.info(f"Processing document: {filename}")
        logger.info(f"=" * 80)

        results = {
            "filename": filename,
            "status": "failed",
            "chunks_created": 0,
            "chunks_uploaded": 0,
            "error": None,
        }

        try:
            # Step 1: Download file (if not provided)
            if file_bytes is None:
                logger.info("Step 1/8: Downloading from Blob Storage...")
                file_bytes = self.blob_service.download_file(filename)
                logger.info(f"Downloaded {len(file_bytes)} bytes")
            else:
                logger.info(f"Step 1/8: Using provided file bytes ({len(file_bytes)} bytes)")

            # Step 2: Determine document type and processing path
            doc_type = self._get_document_type(filename)
            processing_path = self._get_processing_path(doc_type)

            logger.info(
                f"Document type: {doc_type.value}, Processing path: {processing_path}"
            )

            # Step 3-8: Process based on path
            if processing_path == "doc_analysis":
                search_docs = self._process_doc_analysis_path(filename, file_bytes, doc_type)
            else:
                search_docs = self._process_langchain_path(filename, file_bytes, doc_type)

            # Step 9: Upload to Azure AI Search (optional)
            upload_stats = {"successful": 0, "failed": 0, "success_rate": 0.0}
            if not skip_index:
                logger.info(
                    f"Step 8/8: Uploading {len(search_docs)} documents to Azure AI Search..."
                )
                upload_stats = self.search_service.upload_documents_batch(search_docs)

            # Update results
            results["status"] = "success"
            results["chunks_created"] = len(search_docs)
            results["chunks_uploaded"] = upload_stats["successful"]
            results["upload_stats"] = upload_stats
            # Include transformed documents for optional JSON export
            try:
                results["search_documents"] = [doc.to_dict() for doc in search_docs]
            except Exception:
                # Fallback in case serialization fails unexpectedly
                results["search_documents"] = []

            logger.info(f"=" * 80)
            logger.info(f"Successfully processed {filename}")
            logger.info(f"   Chunks created: {results['chunks_created']}")
            logger.info(f"   Chunks uploaded: {results['chunks_uploaded']}")
            logger.info(f"=" * 80)

            return results

        except Exception as e:
            error_msg = f"Failed to process {filename}: {str(e)}"
            logger.error(error_msg)
            results["error"] = str(e)
            return results

    def _process_doc_analysis_path(
        self, filename: str, file_bytes: bytes, doc_type: DocumentType
    ) -> list[SearchDocument]:
        """
        Process document using Document Intelligence path.

        For: PDFs, DOCX, XLSX, PPTX, images

        Args:
            filename: Document filename
            file_bytes: File content
            doc_type: Document type

        Returns:
            list[SearchDocument]: Documents ready for indexing
        """
        # Step 2: Document Intelligence extraction
        logger.info("Step 2/8: Extracting content with Document Intelligence...")
        enriched_doc = self.doc_intel_service.analyze_document(file_bytes, filename, doc_type)
        logger.info(
            f"Extracted {enriched_doc.total_pages} pages, "
            f"{len(enriched_doc.images)} images, "
            f"{sum(p.table_count for p in enriched_doc.pages)} tables"
        )

        # Step 3: Image processing
        if self.config.image_processing.extract_embedded_images and enriched_doc.images:
            logger.info(
                f"Step 3/8: Processing {len(enriched_doc.images)} images (OCR + descriptions)..."
            )
            enriched_doc = self.image_service.process_document_images(enriched_doc, file_bytes)
            logger.info("Processed images")
        else:
            logger.info("Step 3/8: Skipping image processing (disabled or no images)")

        # Step 4: File summary generation
            logger.info("Step 4/8: Generating file summary...")
            file_summary = self.summary_service.generate_summary(enriched_doc)
            logger.info(f"Generated summary ({len(file_summary)} chars)")

        # Step 5: Chunking
        logger.info("Step 5/8: Chunking document...")
        chunks = self.doc_analysis_chunker.chunk_document(enriched_doc)
        logger.info(f"Created {len(chunks)} chunks")

        # Step 6: Metadata enrichment
        logger.info("Step 6/8: Enriching with CSV metadata...")
        doc_metadata = self.metadata_service.lookup_metadata(filename)
        logger.info(
            f"Found metadata: library={doc_metadata.library_name_en}, "
            f"category={doc_metadata.category_name_en}"
        )

        # Step 7: Generate embeddings and create SearchDocuments
        logger.info("Step 7/8: Generating embeddings...")
        search_docs = self._create_search_documents(
            enriched_doc, chunks, file_summary, doc_metadata
        )
        logger.info(f"Created {len(search_docs)} search documents with embeddings")

        return search_docs

    def _process_langchain_path(
        self, filename: str, file_bytes: bytes, doc_type: DocumentType
    ) -> list[SearchDocument]:
        """
        Process text files using LangChain path.

        For: TXT, MD, CSV, JSON

        Args:
            filename: Document filename
            file_bytes: File content
            doc_type: Document type

        Returns:
            list[SearchDocument]: Documents ready for indexing
        """
        # Step 2-3: Skip Document Intelligence and Image Processing
        logger.info("Step 2-3/8: Skipping Document Intelligence and Image Processing (text file)")

        # Create minimal EnrichedDocument
        content = file_bytes.decode("utf-8")
        enriched_doc = EnrichedDocument(
            filename=filename,
            title=Path(filename).stem,
            total_pages=1,
            pages=[],
            markdown=content,
            images=[],
            doc_type=doc_type,
        )

        # Step 4: File summary generation
        logger.info("Step 4/8: Generating file summary...")
        file_summary = self.summary_service.generate_summary(enriched_doc)
        logger.info(f"Generated summary ({len(file_summary)} chars)")

        # Step 5: Chunking
        logger.info("Step 5/8: Chunking document...")
        chunks = self.langchain_chunker.chunk_document(enriched_doc)
        logger.info(f"Created {len(chunks)} chunks")

        # Step 6: Metadata enrichment
        logger.info("Step 6/8: Enriching with CSV metadata...")
        doc_metadata = self.metadata_service.lookup_metadata(filename)
        logger.info(
            f"Found metadata: library={doc_metadata.library_name_en}, "
            f"category={doc_metadata.category_name_en}"
        )

        # Step 7: Generate embeddings and create SearchDocuments
        logger.info("Step 7/8: Generating embeddings...")
        search_docs = self._create_search_documents(
            enriched_doc, chunks, file_summary, doc_metadata
        )
        logger.info(f"Created {len(search_docs)} search documents with embeddings")

        return search_docs

    def _create_search_documents(
        self, enriched_doc: EnrichedDocument, chunks: list[dict], file_summary: str, doc_metadata
    ) -> list[SearchDocument]:
        """
        Create SearchDocuments with embeddings.

        Args:
            enriched_doc: Enriched document
            chunks: Document chunks
            file_summary: File summary text
            doc_metadata: Document metadata from CSV

        Returns:
            list[SearchDocument]: Documents ready for indexing
        """
        search_docs = []

        # Generate chunk embeddings in batch
        chunk_texts = [chunk["content"] for chunk in chunks]
        chunk_embeddings = self.openai_service.generate_embeddings_batch(chunk_texts)

        # Prepare metadata_file (same for all chunks from this file)
        metadata_file_dict = {
            "filename": enriched_doc.filename,
            "title_en": doc_metadata.title_name_en,
            "title_tc": doc_metadata.title_name_tc,
            "library_name_en": doc_metadata.library_name_en,
            "library_name_tc": doc_metadata.library_name_tc,
            "category_name_en": doc_metadata.category_name_en,
            "category_name_tc": doc_metadata.category_name_tc,
            "document_id": doc_metadata.document_id,
            "item_url": doc_metadata.item_url,
            "total_pages": enriched_doc.total_pages,
            "doc_type": enriched_doc.doc_type.value,
        }

        # Merge any additional metadata fields (e.g., from Excel) into the top-level
        try:
            for k, v in (doc_metadata.additional_fields or {}).items():
                # Avoid overwriting existing keys; only add missing ones
                if k not in metadata_file_dict:
                    metadata_file_dict[k] = v
        except Exception:
            # In case additional_fields is not present or unexpected type
            pass

        import json

        metadata_file_json = json.dumps(metadata_file_dict, ensure_ascii=False)

        # Create SearchDocument for each chunk
        for idx, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            chunk_meta = chunk["metadata"]

            # Prepare metadata_chunk
            metadata_chunk_dict = {
                "page_number": chunk_meta.get("page_number"),
                "page_numbers": chunk_meta.get("page_numbers", []),
                "section": chunk_meta.get("section", ""),
                "has_images": chunk_meta.get("has_images", False),
                "image_count": chunk_meta.get("image_count", 0),
                "image_types": chunk_meta.get("image_types", []),
                "chunk_index": chunk_meta.get("chunk_index", idx),
                "total_chunks": chunk_meta.get("total_chunks", len(chunks)),
            }

            metadata_chunk_json = json.dumps(metadata_chunk_dict, ensure_ascii=False)

            # Create unique ID: filename_pageX_chunkY
            page_str = f"p{chunk_meta.get('page_number', 1)}"
            doc_id = f"{Path(enriched_doc.filename).stem}_{page_str}_c{idx}"

            # Generate an LLM-based page description for this chunk
            try:
                page_description = self.openai_service.summarize_text(
                    chunk["content"], max_chars=300
                )
            except Exception:
                page_description = file_summary  # fallback to file-level summary

            # Embed the page description
            try:
                page_summary_embedding = self.openai_service.generate_embedding(page_description)
            except Exception:
                page_summary_embedding = self.openai_service.generate_embedding(file_summary)

            # Create SearchDocument
            search_doc = SearchDocument(
                id=doc_id,
                filename=enriched_doc.filename,
                file_summary=page_description,
                file_summary_chunk=page_summary_embedding,
                metadata_file=metadata_file_json,
                content_chunk=chunk["content"],
                metadata_chunk=metadata_chunk_json,
                content_chunk_dim=embedding,
            )

            search_docs.append(search_doc)

        return search_docs

    def _get_document_type(self, filename: str) -> DocumentType:
        """
        Determine document type from filename extension.

        Args:
            filename: Document filename

        Returns:
            DocumentType: Document type enum
        """
        ext = Path(filename).suffix.lower().lstrip(".")

        type_map = {
            "pdf": DocumentType.PDF,
            "docx": DocumentType.DOCX,
            "doc": DocumentType.DOCX,
            "xlsx": DocumentType.XLSX,
            "xls": DocumentType.XLSX,
            "pptx": DocumentType.PPTX,
            "ppt": DocumentType.PPTX,
            "txt": DocumentType.TXT,
            "md": DocumentType.MD,
            "csv": DocumentType.CSV,
            "json": DocumentType.JSON,
            "png": DocumentType.IMAGE,
            "jpg": DocumentType.IMAGE,
            "jpeg": DocumentType.IMAGE,
        }

        return type_map.get(ext, DocumentType.PDF)

    def _get_processing_path(self, doc_type: DocumentType) -> str:
        """
        Determine processing path based on document type.

        Args:
            doc_type: Document type

        Returns:
            str: "doc_analysis" or "langchain"
        """
        doc_analysis_types = [
            DocumentType.PDF,
            DocumentType.DOCX,
            DocumentType.XLSX,
            DocumentType.PPTX,
            DocumentType.IMAGE,
        ]

        if doc_type in doc_analysis_types:
            return "doc_analysis"
        else:
            return "langchain"

    def process_batch(self, filenames: list[str]) -> dict[str, Any]:
        """
        Process multiple documents in batch.

        Args:
            filenames: List of filenames to process

        Returns:
            dict: Batch processing results

        Example:
            >>> processor = DocumentProcessor(config)
            >>> filenames = blob_service.list_supported_documents()
            >>> results = processor.process_batch(filenames)
            >>> print(f"Processed: {results['successful']}/{results['total']}")
        """
        logger.info(f"Starting batch processing of {len(filenames)} documents")

        total = len(filenames)
        successful = 0
        failed = 0
        results_detail = []

        for idx, filename in enumerate(filenames, 1):
            logger.info(f"\n[{idx}/{total}] Processing: {filename}")

            try:
                result = self.process_document(filename)
                if result["status"] == "success":
                    successful += 1
                else:
                    failed += 1
                results_detail.append(result)

            except Exception as e:
                logger.error(f"Exception processing {filename}: {str(e)}")
                failed += 1
                results_detail.append(
                    {
                        "filename": filename,
                        "status": "failed",
                        "error": str(e),
                    }
                )

        summary = {
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "results": results_detail,
        }

        logger.info(f"\n" + "=" * 80)
        logger.info(f"Batch Processing Complete")
        logger.info(f"Total: {total}, Successful: {successful}, Failed: {failed}")
        logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"=" * 80)

        return summary

    def close(self):
        """Close all service connections."""
        logger.info("Closing DocumentProcessor services...")
        try:
            close_fn = getattr(self.search_service, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception as e:
            logger.warning(f"SearchService close raised: {type(e).__name__}: {e}")
        logger.info("DocumentProcessor closed")


# Example usage
if __name__ == "__main__":
    import sys

    from src.etl.config import get_config

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        config = get_config()
        processor = DocumentProcessor(config)

        print("Document Processor Orchestrator")
        print(f"Blob container: {config.azure_blob.container_name}")
        print(f"Search index: {config.azure_search.index_name}")

        # Test processing
        if len(sys.argv) > 1:
            filename = sys.argv[1]
            print(f"\nProcessing single file: {filename}")
            results = processor.process_document(filename)

            if results["status"] == "success":
                print(f"\nSuccess")
                print(f"   Chunks created: {results['chunks_created']}")
                print(f"   Chunks uploaded: {results['chunks_uploaded']}")
            else:
                print(f"\nFailed: {results['error']}")
        else:
            print("\nUsage:")
            print("  Single file: python document_processor.py <filename>")
            print("  Batch: Modify script to call process_batch()")

        processor.close()

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
