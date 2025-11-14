"""
Azure Document Intelligence service for document extraction.

Handles document analysis using Azure AI Document Intelligence API.
Extracts text, tables, sections, and image locations from documents.
"""

import logging
from io import BytesIO
from typing import BinaryIO

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.pipeline.transport import RequestsTransport
from azure.core.credentials import AzureKeyCredential

from src.config import DocumentIntelligenceConfig
from src.models.document_models import DocumentType, EnrichedDocument, ImageInfo, PageContent

logger = logging.getLogger(__name__)


class DocumentIntelligenceService:
    """
    Service for analyzing documents using Azure Document Intelligence.

    Uses the prebuilt-layout model to extract structured content including
    text, tables, sections, and image locations.
    """

    def __init__(self, config: DocumentIntelligenceConfig):
        """
        Initialize Document Intelligence service.

        Args:
            config: Document Intelligence configuration
        """
        self.config = config
        # Configure transport timeouts to accommodate long-running analyses
        transport = RequestsTransport(
            connection_timeout=config.timeout_seconds,
            read_timeout=config.timeout_seconds,
        )

        self.client = DocumentAnalysisClient(
            endpoint=config.endpoint,
            credential=AzureKeyCredential(config.key),
            transport=transport,
        )
        logger.info(f"Initialized DocumentIntelligenceService: {config.endpoint}")

    def analyze_document(
        self, document: bytes | BinaryIO, filename: str, doc_type: DocumentType
    ) -> EnrichedDocument:
        """
        Analyze a document and extract structured content.

        Args:
            document: Document content (bytes or file-like object)
            filename: Original filename
            doc_type: Type of document

        Returns:
            EnrichedDocument: Extracted and structured document content

        Example:
            >>> doc_intel = DocumentIntelligenceService(config)
            >>> with open("policy.pdf", "rb") as f:
            >>>     result = doc_intel.analyze_document(f, "policy.pdf", DocumentType.PDF)
            >>> print(f"Extracted {result.total_pages} pages")
        """
        logger.info(f"Analyzing document: {filename} (type: {doc_type.value})")

        try:
            # Convert to BytesIO if needed
            if isinstance(document, bytes):
                document = BytesIO(document)

            # Call Document Intelligence API with simple retry on timeouts
            result = self._analyze_with_retry(document)

            # Extract pages
            pages = self._extract_pages(result)

            # Extract images (locations only, actual extraction done later)
            images = self._extract_image_locations(result)

            # Convert to markdown
            markdown = self._convert_to_markdown(result, pages)

            # Get title (from first paragraph or filename)
            title = self._extract_title(result, filename)

            enriched_doc = EnrichedDocument(
                filename=filename,
                title=title,
                total_pages=len(pages),
                pages=pages,
                markdown=markdown,
                images=images,
                doc_type=doc_type,
                processing_metadata={
                    "api_version": self.config.api_version,
                    "model": "prebuilt-layout",
                    "page_count": len(pages),
                    "table_count": sum(p.table_count for p in pages),
                    "image_count": len(images),
                },
            )

            logger.info(
                f"Successfully analyzed {filename}: "
                f"{enriched_doc.total_pages} pages, "
                f"{len(images)} images, "
                f"{sum(p.table_count for p in pages)} tables"
            )

            return enriched_doc

        except Exception as e:
            logger.error(f"Failed to analyze document {filename}: {str(e)}")
            raise

    def _analyze_with_retry(self, document: BinaryIO):
        """Run analyze_document with basic retry for timeout errors."""
        import time
        attempts = getattr(self.config, "retry_attempts", 0) or 0
        delay = getattr(self.config, "retry_delay_seconds", 5) or 5
        last_err: Exception | None = None

        for attempt in range(0, attempts + 1):
            try:
                poller = self.client.begin_analyze_document("prebuilt-layout", document)
                return poller.result()
            except Exception as e:
                last_err = e
                msg = str(e)
                is_timeout = ("Timeout" in msg) or ("timed out" in msg) or ("ReadTimeout" in msg)
                is_conn_reset = ("Connection aborted" in msg) or ("ConnectionResetError" in msg)
                logger.warning(
                    f"Document Intelligence analyze attempt {attempt + 1}/{attempts + 1} failed: {msg}"
                )
                if attempt < attempts and (is_timeout or is_conn_reset):
                    time.sleep(delay)
                    # Rewind stream if possible before retry
                    try:
                        document.seek(0)
                    except Exception:
                        pass
                    continue
                # Non-timeout or final attempt: raise
                raise e

        # Should not reach here; raise last error
        if last_err:
            raise last_err
        raise RuntimeError("Failed to analyze document: unknown error")

    def _extract_pages(self, result) -> list[PageContent]:
        """Extract content from each page."""
        pages: list[PageContent] = []

        for page in result.pages:
            # Extract text content
            content_parts = []

            # Get lines in reading order
            if hasattr(page, "lines") and page.lines:
                for line in page.lines:
                    content_parts.append(line.content)

            content = "\n".join(content_parts)

            # Check for tables on this page
            tables_on_page = [
                table
                for table in (result.tables or [])
                if any(cell.bounding_regions and cell.bounding_regions[0].page_number == page.page_number for cell in table.cells)
            ]

            page_content = PageContent(
                page_number=page.page_number,
                content=content,
                has_tables=len(tables_on_page) > 0,
                table_count=len(tables_on_page),
            )

            pages.append(page_content)

        return pages

    def _extract_image_locations(self, result) -> list[ImageInfo]:
        """
        Extract image locations from document.

        Note: This only extracts metadata about where images are located.
        Actual image extraction and processing happens in image_processing_service.py
        """
        images: list[ImageInfo] = []

        if not hasattr(result, "figures") or not result.figures:
            return images

        for idx, figure in enumerate(result.figures):
            # Get bounding region (page and position)
            page_number = None
            position = None

            if figure.bounding_regions:
                page_number = figure.bounding_regions[0].page_number
                polygon = figure.bounding_regions[0].polygon
                if polygon and len(polygon) >= 4:
                    # Calculate bounding box from polygon
                    x_coords = [p.x for p in polygon]
                    y_coords = [p.y for p in polygon]
                    position = {
                        "x": min(x_coords),
                        "y": min(y_coords),
                        "width": max(x_coords) - min(x_coords),
                        "height": max(y_coords) - min(y_coords),
                    }

            image_info = ImageInfo(
                image_id=f"img_{idx}",
                page_number=page_number,
                position=position,
                # OCR and description will be added by image_processing_service
            )

            images.append(image_info)

        return images

    def _convert_to_markdown(self, result, pages: list[PageContent]) -> str:
        """
        Convert document content to markdown format.

        Preserves structure including headers, paragraphs, and tables.
        """
        markdown_parts = []

        # Add title/document name if available
        # (Will be refined later after image processing)

        # Process paragraphs
        if hasattr(result, "paragraphs") and result.paragraphs:
            for para in result.paragraphs:
                # Check if it's a heading
                if hasattr(para, "role") and para.role in ["title", "sectionHeading"]:
                    # Add as markdown heading
                    level = 1 if para.role == "title" else 2
                    markdown_parts.append(f"{'#' * level} {para.content}\n")
                else:
                    # Regular paragraph
                    markdown_parts.append(f"{para.content}\n")

        # Process tables
        if hasattr(result, "tables") and result.tables:
            for table in result.tables:
                markdown_parts.append(self._table_to_markdown(table))

        return "\n".join(markdown_parts)

    def _table_to_markdown(self, table) -> str:
        """Convert a table to markdown format."""
        if not table.cells:
            return ""

        # Determine table dimensions
        max_row = max(cell.row_index for cell in table.cells) + 1
        max_col = max(cell.column_index for cell in table.cells) + 1

        # Create table grid
        grid = [[None for _ in range(max_col)] for _ in range(max_row)]

        # Fill grid
        for cell in table.cells:
            grid[cell.row_index][cell.column_index] = cell.content or ""

        # Convert to markdown
        markdown_lines = []

        # Header row
        if max_row > 0:
            header = "| " + " | ".join(str(cell or "") for cell in grid[0]) + " |"
            markdown_lines.append(header)
            separator = "|" + "|".join([" --- " for _ in range(max_col)]) + "|"
            markdown_lines.append(separator)

        # Data rows
        for row in grid[1:]:
            row_md = "| " + " | ".join(str(cell or "") for cell in row) + " |"
            markdown_lines.append(row_md)

        markdown_lines.append("")  # Empty line after table
        return "\n".join(markdown_lines)

    def _extract_title(self, result, filename: str) -> str:
        """Extract document title from content or use filename."""
        # Try to get title from first heading
        if hasattr(result, "paragraphs") and result.paragraphs:
            for para in result.paragraphs:
                if hasattr(para, "role") and para.role == "title":
                    return para.content

            # Fallback to first paragraph if short enough
            first_para = result.paragraphs[0].content
            if len(first_para) < 100:
                return first_para

        # Fallback to filename without extension
        from pathlib import Path

        return Path(filename).stem.replace("_", " ").replace("-", " ").title()

    def close(self):
        """Close the Document Intelligence client."""
        # Azure SDK handles cleanup automatically
        logger.info("Closed DocumentIntelligenceService")


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path

    from src.config import get_config

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load config
    try:
        config = get_config()
        doc_intel = DocumentIntelligenceService(config.doc_intelligence)

        # Test with a file (if provided)
        if len(sys.argv) > 1:
            file_path = Path(sys.argv[1])
            if file_path.exists():
                print(f"Analyzing document: {file_path.name}\n")

                with open(file_path, "rb") as f:
                    # Determine doc type from extension
                    ext = file_path.suffix.lower().lstrip(".")
                    doc_type = DocumentType.PDF if ext == "pdf" else DocumentType.DOCX

                    result = doc_intel.analyze_document(f, file_path.name, doc_type)

                print(f"✅ Successfully analyzed document:")
                print(f"   Title: {result.title}")
                print(f"   Pages: {result.total_pages}")
                print(f"   Images: {len(result.images)}")
                print(f"   Tables: {sum(p.table_count for p in result.pages)}")
                print(f"\n📝 First 500 chars of markdown:")
                print(result.markdown[:500])
            else:
                print(f"File not found: {file_path}")
                sys.exit(1)
        else:
            print("Usage: python doc_intel_service.py <path_to_document>")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
