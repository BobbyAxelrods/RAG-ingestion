"""
Main Orchestrator for Document Indexing Pipeline.

Provides CLI interface for running the complete document processing pipeline.

Usage:
    # Process single file
    python src/main.py process --file policy.pdf

    # Process all files in blob container
    python src/main.py process --all

    # Create/recreate search index
    python src/main.py create-index --recreate

    # Validate configuration
    python src/main.py validate

    # Show statistics
    python src/main.py stats
"""

import argparse
import logging
import sys
from pathlib import Path

from src.etl.config import get_config
from src.etl.processors.document_processor import DocumentProcessor
from src.etl.services.blob_service import BlobStorageService
from src.etl.services.metadata_enrichment_service import MetadataEnrichmentService
from src.etl.services.search_service import SearchService
from src.etl.services.etl_search_service import ETLSearchService

# Ensure logs directory exists before configuring logging
Path("logs").mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def create_index(recreate: bool = False):
    """
    Create Azure AI Search index.

    Args:
        recreate: If True, delete and recreate index
    """
    print("🔧 Creating Azure AI Search Index")
    print("=" * 80)

    try:
        config = get_config()
        search_service = SearchService(config.azure_search, config.azure_openai)

        print(f"Endpoint: {config.azure_search.endpoint}")
        print(f"Index: {config.azure_search.index_name}")

        if search_service.create_index(recreate=recreate):
            print("✅ Index created successfully!")

            # Get document count
            count = search_service.get_document_count()
            print(f"📊 Current document count: {count}")
        else:
            print("❌ Failed to create index")
            sys.exit(1)

        search_service.close()

    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def create_etl_index(recreate: bool = False):
    """Create Azure AI Search ETL (hierarchical) index."""
    print("🔧 Creating Azure AI Search ETL Index")
    print("=" * 80)

    try:
        config = get_config()
        etl_service = ETLSearchService(config.azure_search, config.azure_openai)

        print(f"Endpoint: {config.azure_search.endpoint}")
        print(f"ETL Index: {etl_service.index_name}")

        if etl_service.create_index(recreate=recreate):
            print("✅ ETL index created successfully!")
            count = etl_service.get_document_count()
            print(f"📊 Current ETL document count: {count}")
        else:
            print("❌ Failed to create ETL index")
            sys.exit(1)

        etl_service.close()

    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def process_file(filename: str, output_json: str | None = None, no_index: bool = False, local: bool = False, local_dir: str | None = None):
    """
    Process a single file.

    Args:
        filename: Filename to process
    """
    print(f"Processing Single File: {filename}")
    print("=" * 80)

    try:
        config = get_config()
        # Disable SearchService when skipping index or generating JSON only
        enable_search = not (no_index or (output_json is not None))
        processor = DocumentProcessor(config, enable_search=enable_search)

        # Resolve local file bytes if requested
        file_bytes = None
        display_name = filename
        if local:
            file_path = Path(filename)
            if not file_path.is_absolute() and local_dir:
                file_path = Path(local_dir) / filename
            # If still not absolute, resolve relative to cwd
            file_path = file_path.resolve()
            if not file_path.exists():
                print(f"\nERROR: Local file not found: {file_path}")
                sys.exit(1)
            file_bytes = file_path.read_bytes()
            display_name = file_path.name  # use basename for metadata CSV lookup

        results = processor.process_document(display_name, file_bytes=file_bytes, skip_index=no_index or (output_json is not None))

        if results["status"] == "success":
            print("\nProcessing Complete")
            print(f"   Chunks created: {results['chunks_created']}")
            print(f"   Chunks uploaded: {results['chunks_uploaded']}")
            print(f"   Upload success rate: {results['upload_stats']['success_rate']:.1f}%")

            # Write transformation output to JSON if requested
            if output_json:
                try:
                    out_path = Path(output_json)
                    original_path = filename if local else None
                    # Build ETL JSON from SearchDocuments
                    from src.etl.models.document_models import SearchDocument
                    from src.etl.services.extraction_writer_service import ExtractionWriter

                    search_docs = [SearchDocument(**sd) for sd in results.get("search_documents", [])]
                    ExtractionWriter().write_extraction_json(
                        display_name,
                        search_docs,
                        out_path,
                        original_path=original_path,
                    )
                    print(f"\nWrote ETL JSON: {out_path}")
                except Exception as e:
                    print(f"\nWARNING: Failed to write JSON: {str(e)}")
        else:
            print(f"\nProcessing Failed: {results['error']}")
            sys.exit(1)

        processor.close()

    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def process_all_files():
    """Process all supported files in blob container."""
    print("📦 Processing All Files from Blob Storage")
    print("=" * 80)

    try:
        config = get_config()

        # Initialize services
        blob_service = BlobStorageService(config.azure_blob)
        processor = DocumentProcessor(config)

        # List supported files
        print("Listing files from blob storage...")
        filenames = blob_service.list_supported_documents()

        if not filenames:
            print("⚠️ No supported documents found in blob storage")
            sys.exit(0)

        print(f"Found {len(filenames)} supported documents\n")

        # Confirm with user
        print("Files to process:")
        for idx, filename in enumerate(filenames, 1):
            print(f"  {idx}. {filename}")

        response = input("\nProceed with batch processing? (y/n): ")
        if response.lower() != "y":
            print("❌ Cancelled")
            sys.exit(0)

        # Process batch
        print("\nStarting batch processing...")
        results = processor.process_batch(filenames)

        # Display summary
        print("\n" + "=" * 80)
        print("📊 Batch Processing Summary")
        print("=" * 80)
        print(f"Total files: {results['total']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"Success rate: {results['success_rate']:.1f}%")

        if results["failed"] > 0:
            print("\n❌ Failed files:")
            for result in results["results"]:
                if result["status"] == "failed":
                    print(f"  - {result['filename']}: {result.get('error', 'Unknown error')}")

        processor.close()

    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def validate_config():
    """Validate configuration and services."""
    print("✅ Validating Configuration")
    print("=" * 80)

    try:
        config = get_config()

        print("Configuration loaded successfully!")
        print("\n📋 Service Configurations:")
        print(f"  - Azure Blob Storage: {config.azure_blob.container_name}")
        print(f"  - Document Intelligence: {config.doc_intelligence.endpoint}")
        print(f"  - Azure OpenAI: {config.azure_openai.endpoint}")
        print(f"  - Azure AI Search: {config.azure_search.endpoint}")
        print(f"  - Metadata CSV: {config.metadata.csv_path}")

        # Validate CSV
        print("\n📊 Validating Metadata CSV...")
        metadata_service = MetadataEnrichmentService(config.metadata)
        is_valid, issues = metadata_service.validate_csv()

        if is_valid:
            print("✅ CSV is valid")
            stats = metadata_service.get_statistics()
            print(f"   Total files: {stats['total_files']}")
            print(f"   Unique libraries: {stats['unique_libraries']}")
            print(f"   Unique categories: {stats['unique_categories']}")
        else:
            print("⚠️ CSV has issues:")
            for issue in issues:
                print(f"   - {issue}")

        # Check search index
        print("\n🔍 Checking Search Index...")
        search_service = SearchService(config.azure_search, config.azure_openai)

        if search_service.index_exists():
            count = search_service.get_document_count()
            print(f"✅ Index exists: {config.azure_search.index_name}")
            print(f"   Document count: {count}")
        else:
            print(f"⚠️ Index does not exist: {config.azure_search.index_name}")
            print("   Run 'python src/main.py create-index' to create it")

        search_service.close()

        print("\n✅ Validation complete!")

    except Exception as e:
        print(f"❌ Validation failed: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def show_statistics():
    """Show statistics about indexed documents."""
    print("📊 Index Statistics")
    print("=" * 80)

    try:
        config = get_config()

        # Metadata statistics
        print("CSV Metadata:")
        metadata_service = MetadataEnrichmentService(config.metadata)
        stats = metadata_service.get_statistics()

        print(f"  Total files in CSV: {stats['total_files']}")
        print(f"  Files with images: {stats['files_with_images']}")
        print(f"  Total images: {stats['total_images']}")

        print(f"\n  Libraries ({stats['unique_libraries']}):")
        for library, count in list(stats["libraries"].items())[:5]:
            print(f"    - {library}: {count} files")

        print(f"\n  Categories ({stats['unique_categories']}):")
        for category, count in list(stats["categories"].items())[:5]:
            print(f"    - {category}: {count} files")

        # Search index statistics
        print("\nSearch Index:")
        search_service = SearchService(config.azure_search, config.azure_openai)

        if search_service.index_exists():
            count = search_service.get_document_count()
            print(f"  Index: {config.azure_search.index_name}")
            print(f"  Total chunks: {count}")
        else:
            print(f"  Index does not exist: {config.azure_search.index_name}")

        search_service.close()

    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Document Indexing Pipeline for Prudential Hong Kong RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process documents")
    process_group = process_parser.add_mutually_exclusive_group(required=True)
    process_group.add_argument("--file", type=str, help="Process single file")
    process_group.add_argument(
        "--all", action="store_true", help="Process all files in blob storage"
    )

    # Optional controls
    process_parser.add_argument("--output-json", type=str, help="Write transformation output to JSON path")
    process_parser.add_argument("--no-index", action="store_true", help="Skip uploading documents to Azure AI Search")
    process_parser.add_argument("--local", action="store_true", help="Read the file from local filesystem instead of Blob")
    process_parser.add_argument("--local-dir", type=str, help="Directory containing the local file (used when --local is set)")

    # Create index command
    index_parser = subparsers.add_parser("create-index", help="Create search index")
    index_parser.add_argument(
        "--recreate", action="store_true", help="Delete and recreate index if it exists"
    )

    # Create ETL index command
    etl_index_parser = subparsers.add_parser("create-etl-index", help="Create ETL (hierarchical) search index")
    etl_index_parser.add_argument(
        "--recreate", action="store_true", help="Delete and recreate ETL index if it exists"
    )

    # Upload ETL JSON command
    upload_etl_parser = subparsers.add_parser("upload-etl-json", help="Upload a single ETL JSON file to ETL index")
    upload_etl_parser.add_argument("--file", type=str, required=True, help="Path to ETL JSON file")

    # Validate command
    subparsers.add_parser("validate", help="Validate configuration and services")

    # Stats command
    subparsers.add_parser("stats", help="Show statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    # Route to appropriate function
    if args.command == "process":
        if args.file:
            process_file(args.file, output_json=args.output_json, no_index=args.no_index, local=args.local, local_dir=args.local_dir)
        elif args.all:
            process_all_files()
    elif args.command == "create-index":
        create_index(recreate=args.recreate)
    elif args.command == "create-etl-index":
        create_etl_index(recreate=args.recreate)
    elif args.command == "validate":
        validate_config()
    elif args.command == "stats":
        show_statistics()
    elif args.command == "upload-etl-json":
        config = get_config()
        etl_service = ETLSearchService(config.azure_search, config.azure_openai)
        json_path = Path(getattr(args, "file"))
        if not json_path.exists():
            print(f"❌ ETL JSON file not found: {json_path}")
            sys.exit(1)
        ok = etl_service.upload_etl_json(json_path)
        print("✅ Uploaded" if ok else "❌ Upload failed")
        etl_service.close()


if __name__ == "__main__":
    main()
