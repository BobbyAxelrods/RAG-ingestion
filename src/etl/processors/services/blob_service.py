"""
Azure Blob Storage service for document ingestion pipeline.

Handles listing, downloading, and managing files from Azure Blob Storage.
"""

import logging
from io import BytesIO
from typing import Generator

from azure.storage.blob import BlobServiceClient, ContainerClient

from src.config import AzureBlobConfig
from src.models.document_models import BlobFile

logger = logging.getLogger(__name__)


class BlobStorageService:
    """
    Service for interacting with Azure Blob Storage.

    Provides methods to list and download files from a container.
    """

    def __init__(self, config: AzureBlobConfig):
        """
        Initialize Blob Storage service.

        Args:
            config: Azure Blob Storage configuration
        """
        self.config = config
        self.blob_service_client: BlobServiceClient | None = None
        self.container_client: ContainerClient | None = None

        # Lazy init to support local-only runs without blob config
        if config.connection_string and config.container_name:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                config.connection_string
            )
            self.container_client = self.blob_service_client.get_container_client(
                config.container_name
            )
            logger.info(
                f"Initialized BlobStorageService for container: {config.container_name}"
            )
        else:
            logger.info(
                "BlobStorageService initialized in disabled mode (missing connection string/container)."
            )

    def list_files(
        self, prefix: str | None = None, file_extensions: list[str] | None = None
    ) -> Generator[BlobFile, None, None]:
        """
        List all files in the container.

        Args:
            prefix: Optional prefix to filter files (e.g., "folder1/")
            file_extensions: Optional list of extensions to filter (e.g., [".pdf", ".docx"])

        Yields:
            BlobFile: Information about each blob file

        Example:
            >>> blob_service = BlobStorageService(config)
            >>> for file in blob_service.list_files(file_extensions=[".pdf"]):
            >>>     print(f"Found PDF: {file.name}")
        """
        logger.info(
            f"Listing files in container: {self.config.container_name} "
            f"(prefix={prefix}, extensions={file_extensions})"
        )

        if not self.container_client:
            raise RuntimeError("Blob storage is not configured.")
        blob_list = self.container_client.list_blobs(name_starts_with=prefix)
        file_count = 0

        for blob in blob_list:
            # Skip directories (blobs ending with /)
            if blob.name.endswith("/"):
                continue

            # Filter by extension if specified
            if file_extensions:
                if not any(blob.name.lower().endswith(ext.lower()) for ext in file_extensions):
                    continue

            blob_file = BlobFile(
                name=blob.name,
                size=blob.size,
                last_modified=blob.last_modified,
                content_type=blob.content_settings.content_type
                if blob.content_settings
                else None,
            )

            file_count += 1
            logger.debug(f"Found file: {blob_file.name} ({blob_file.size} bytes)")
            yield blob_file

        logger.info(f"Found {file_count} files")

    def download_file(self, blob_name: str) -> bytes:
        """
        Download a file from blob storage.

        Args:
            blob_name: Name of the blob to download

        Returns:
            bytes: File content

        Raises:
            Exception: If download fails

        Example:
            >>> blob_service = BlobStorageService(config)
            >>> content = blob_service.download_file("documents/policy.pdf")
            >>> print(f"Downloaded {len(content)} bytes")
        """
        logger.info(f"Downloading file: {blob_name}")

        try:
            if not self.container_client:
                raise RuntimeError("Blob storage is not configured.")
            blob_client = self.container_client.get_blob_client(blob_name)
            download_stream = blob_client.download_blob()
            content = download_stream.readall()

            logger.info(f"Successfully downloaded {blob_name} ({len(content)} bytes)")
            return content

        except Exception as e:
            logger.error(f"Failed to download {blob_name}: {str(e)}")
            raise

    def download_file_to_stream(self, blob_name: str) -> BytesIO:
        """
        Download a file to a BytesIO stream.

        Useful for passing to other services without writing to disk.

        Args:
            blob_name: Name of the blob to download

        Returns:
            BytesIO: Stream containing file content

        Example:
            >>> blob_service = BlobStorageService(config)
            >>> stream = blob_service.download_file_to_stream("doc.pdf")
            >>> # Pass stream to Document Intelligence
        """
        content = self.download_file(blob_name)
        stream = BytesIO(content)
        stream.name = blob_name  # Set name for reference
        return stream

    def file_exists(self, blob_name: str) -> bool:
        """
        Check if a file exists in the container.

        Args:
            blob_name: Name of the blob to check

        Returns:
            bool: True if file exists, False otherwise
        """
        try:
            if not self.container_client:
                return False
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.get_blob_properties()
            return True
        except Exception:
            return False

    def get_file_info(self, blob_name: str) -> BlobFile:
        """
        Get detailed information about a specific file.

        Args:
            blob_name: Name of the blob

        Returns:
            BlobFile: File information

        Raises:
            Exception: If file doesn't exist
        """
        logger.debug(f"Getting file info for: {blob_name}")

        try:
            if not self.container_client:
                raise RuntimeError("Blob storage is not configured.")
            blob_client = self.container_client.get_blob_client(blob_name)
            properties = blob_client.get_blob_properties()

            return BlobFile(
                name=blob_name,
                size=properties.size,
                last_modified=properties.last_modified,
                content_type=properties.content_settings.content_type
                if properties.content_settings
                else None,
            )

        except Exception as e:
            logger.error(f"Failed to get file info for {blob_name}: {str(e)}")
            raise

    def list_supported_documents(self) -> Generator[BlobFile, None, None]:
        """
        List all supported document types in the container.

        Filters for: .pdf, .docx, .pptx, .txt, .md, .csv, .json,
                     .png, .jpg, .jpeg, .bmp, .tiff

        Yields:
            BlobFile: Information about each supported file

        Example:
            >>> blob_service = BlobStorageService(config)
            >>> for file in blob_service.list_supported_documents():
            >>>     print(f"Processing: {file.name}")
        """
        supported_extensions = [
            ".pdf",
            ".docx",
            ".pptx",
            ".txt",
            ".md",
            ".csv",
            ".json",
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tiff",
        ]

        yield from self.list_files(file_extensions=supported_extensions)

    def close(self):
        """Close the blob service client."""
        if self.blob_service_client:
            self.blob_service_client.close()
            logger.info("Closed BlobStorageService")


# Example usage
if __name__ == "__main__":
    import sys
    from src.config import get_config

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load config
    try:
        config = get_config()
        blob_service = BlobStorageService(config.azure_blob)

        print(f"Listing files in container: {config.azure_blob.container_name}\n")

        # List all supported documents
        for file in blob_service.list_supported_documents():
            print(f"📄 {file.name}")
            print(f"   Size: {file.size:,} bytes")
            print(f"   Type: {file.document_type.value}")
            print(f"   Modified: {file.last_modified}")
            print()

        blob_service.close()

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
