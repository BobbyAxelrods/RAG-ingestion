"""
File Summary Generation Service.

Implements extractive summary generation focusing on:
1. Document purpose extraction (from title, first paragraphs, or headers)
2. Page-by-page key sentence extraction using importance scoring

This is NOT LLM-based summarization - it uses rule-based extractive methods
for cost efficiency and predictability.
"""

import logging
import re
from typing import Any

from src.config import FileSummaryConfig
from src.models.document_models import EnrichedDocument, PageContent

logger = logging.getLogger(__name__)


class SummaryGenerationService:
    """
    Service for generating extractive summaries from documents.

    Uses a two-part approach:
    1. Document Purpose: Why this document exists (1-2 sentences)
    2. Page Summaries: Key sentences from each page using scoring algorithm
    """

    def __init__(self, config: FileSummaryConfig):
        """
        Initialize Summary Generation service.

        Args:
            config: File summary configuration
        """
        self.config = config

        logger.info(
            f"Initialized SummaryGenerationService "
            f"(method={config.summary_method}, max_sentences={config.max_sentences_per_page})"
        )

    def generate_summary(self, enriched_doc: EnrichedDocument) -> str:
        """
        Generate extractive summary for a document.

        Strategy:
        1. Extract document purpose (1-2 sentences)
        2. Extract key sentences from each page
        3. Combine into coherent summary (max length enforced)

        Args:
            enriched_doc: Document processed by Document Intelligence

        Returns:
            str: Document summary

        Example:
            >>> summary_service = SummaryGenerationService(config)
            >>> enriched_doc = doc_intel_service.analyze_document(...)
            >>> summary = summary_service.generate_summary(enriched_doc)
            >>> print(f"Summary: {summary[:200]}...")
        """
        logger.info(
            f"Generating summary for {enriched_doc.filename} "
            f"({enriched_doc.total_pages} pages, {len(enriched_doc.markdown)} chars)"
        )

        try:
            # Step 1: Extract document purpose
            purpose = self._extract_document_purpose(enriched_doc)

            # Step 2: Extract key sentences from pages
            if self.config.include_page_summaries:
                page_summaries = self._extract_page_summaries(enriched_doc)
            else:
                page_summaries = ""

            # Step 3: Combine purpose and page summaries
            summary_parts = []

            if purpose:
                summary_parts.append(f"Purpose: {purpose}")

            if page_summaries:
                summary_parts.append(f"\nKey Points:\n{page_summaries}")

            summary = "\n".join(summary_parts)

            # Step 4: Truncate if needed
            if len(summary) > self.config.max_summary_length:
                summary = summary[: self.config.max_summary_length] + "..."
                logger.debug(f"Truncated summary to {self.config.max_summary_length} chars")

            logger.info(
                f"✅ Generated summary for {enriched_doc.filename} ({len(summary)} chars)"
            )

            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary for {enriched_doc.filename}: {str(e)}")
            # Fallback: use title or filename
            return f"Document: {enriched_doc.title or enriched_doc.filename}"

    def _extract_document_purpose(self, enriched_doc: EnrichedDocument) -> str:
        """
        Extract document purpose using three strategies.

        Strategy 1: Use title if descriptive
        Strategy 2: Extract from first paragraph
        Strategy 3: Look for purpose keywords

        Args:
            enriched_doc: Document to analyze

        Returns:
            str: Document purpose (1-2 sentences)
        """
        # Strategy 1: Use title if it's descriptive (not just filename)
        if enriched_doc.title and len(enriched_doc.title) > 10:
            title_lower = enriched_doc.title.lower()
            # Check if title is descriptive (not just "document" or filename-like)
            if not any(
                x in title_lower
                for x in ["untitled", "document", "pdf", "doc", enriched_doc.filename.lower()]
            ):
                logger.debug("Using title as document purpose")
                return f"This document is about {enriched_doc.title}."

        # Strategy 2: Extract from first paragraph
        first_para = self._get_first_paragraph(enriched_doc.markdown)
        if first_para and len(first_para) > 20:
            # Extract first 1-2 sentences
            sentences = self._split_into_sentences(first_para)
            if sentences:
                purpose = " ".join(sentences[: min(2, len(sentences))])
                logger.debug("Extracted purpose from first paragraph")
                return purpose

        # Strategy 3: Look for purpose-indicating patterns
        purpose_patterns = [
            r"This document (?:describes|outlines|provides|explains) (.+?)(?:\.|$)",
            r"The purpose of this (?:document|guide|policy) is to (.+?)(?:\.|$)",
            r"This (?:guide|manual|policy) (?:covers|addresses|details) (.+?)(?:\.|$)",
        ]

        for pattern in purpose_patterns:
            match = re.search(pattern, enriched_doc.markdown, re.IGNORECASE)
            if match:
                purpose = match.group(0)
                logger.debug("Extracted purpose using pattern matching")
                return purpose

        # Fallback: Use title or filename
        logger.debug("Using fallback purpose (title/filename)")
        return f"Document: {enriched_doc.title or enriched_doc.filename}"

    def _extract_page_summaries(self, enriched_doc: EnrichedDocument) -> str:
        """
        Extract key sentences from each page.

        Uses importance scoring algorithm to select top N sentences per page.

        Args:
            enriched_doc: Document to analyze

        Returns:
            str: Formatted page summaries
        """
        page_summaries = []

        for page in enriched_doc.pages:
            # Skip empty pages
            if not page.content or len(page.content.strip()) < 50:
                continue

            # Extract key sentences for this page
            key_sentences = self._extract_key_sentences(
                page.content, max_sentences=self.config.max_sentences_per_page
            )

            if key_sentences:
                # Format page summary
                page_summary = f"Page {page.page_number}: {' '.join(key_sentences)}"
                page_summaries.append(page_summary)

        return "\n".join(page_summaries)

    def _extract_key_sentences(self, text: str, max_sentences: int = 2) -> list[str]:
        """
        Extract key sentences using importance scoring.

        Scoring criteria (7 factors):
        1. Position (first sentences score higher)
        2. Length (50-150 chars ideal)
        3. Keyword density (financial, insurance, policy, etc.)
        4. Contains numbers/dates
        5. Contains capitalized terms
        6. Sentence completeness
        7. Not a heading or title

        Args:
            text: Text to extract from
            max_sentences: Maximum sentences to extract

        Returns:
            list[str]: Top-scored sentences
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        # Score each sentence
        scored_sentences = []
        for idx, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, idx, len(sentences))
            scored_sentences.append((score, sentence))

        # Sort by score (descending)
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # Return top N sentences (in original order)
        top_sentences = [sent for _, sent in scored_sentences[:max_sentences]]

        # Re-order by original position
        original_order = []
        for sent in top_sentences:
            original_idx = sentences.index(sent)
            original_order.append((original_idx, sent))

        original_order.sort(key=lambda x: x[0])

        return [sent for _, sent in original_order]

    def _score_sentence(self, sentence: str, position: int, total_sentences: int) -> float:
        """
        Score a sentence based on importance criteria.

        Args:
            sentence: Sentence to score
            position: Position in text (0-based)
            total_sentences: Total number of sentences

        Returns:
            float: Importance score (higher = more important)
        """
        score = 0.0

        # Criterion 1: Position score (first sentences are more important)
        position_score = 1.0 - (position / max(total_sentences, 1))
        score += position_score * 3.0

        # Criterion 2: Length score (ideal: 50-150 chars)
        length = len(sentence)
        if 50 <= length <= 150:
            length_score = 2.0
        elif 30 <= length < 50 or 150 < length <= 200:
            length_score = 1.0
        else:
            length_score = 0.5
        score += length_score

        # Criterion 3: Keyword density (insurance domain keywords)
        keywords = [
            "insurance",
            "policy",
            "premium",
            "coverage",
            "benefit",
            "claim",
            "insured",
            "policyholder",
            "payment",
            "terms",
            "conditions",
            "exclusion",
            "rider",
            "sum assured",
            "maturity",
        ]
        sentence_lower = sentence.lower()
        keyword_count = sum(1 for kw in keywords if kw in sentence_lower)
        score += keyword_count * 1.5

        # Criterion 4: Contains numbers/dates
        if re.search(r"\d+", sentence):
            score += 1.0

        # Criterion 5: Contains capitalized terms (proper nouns, important terms)
        capitalized_words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", sentence)
        if len(capitalized_words) >= 2:
            score += 1.0

        # Criterion 6: Sentence completeness (ends with period, not truncated)
        if sentence.strip().endswith("."):
            score += 0.5

        # Criterion 7: Not a heading/title (penalize very short, all-caps, or colon-ending)
        if len(sentence.strip()) < 20:
            score -= 1.0
        if sentence.isupper():
            score -= 2.0
        if sentence.strip().endswith(":"):
            score -= 1.5

        return max(score, 0.0)

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            list[str]: List of sentences
        """
        # Simple sentence splitting (handles most cases)
        # Split on period, question mark, exclamation mark followed by space/newline
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Clean up sentences
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if sent and len(sent) > 10:  # Ignore very short fragments
                cleaned_sentences.append(sent)

        return cleaned_sentences

    def _get_first_paragraph(self, markdown: str) -> str:
        """
        Extract first substantial paragraph from markdown.

        Args:
            markdown: Markdown content

        Returns:
            str: First paragraph or empty string
        """
        # Split by double newlines (paragraph separator)
        paragraphs = re.split(r"\n\n+", markdown)

        for para in paragraphs:
            para = para.strip()

            # Skip headers
            if para.startswith("#"):
                continue

            # Skip very short paragraphs
            if len(para) < 30:
                continue

            # Skip table markers
            if "|" in para and "---" in para:
                continue

            return para

        return ""

    def generate_summary_from_chunks(self, chunks: list[dict[str, Any]]) -> str:
        """
        Generate summary from pre-chunked content.

        Useful when you have chunks but not the original EnrichedDocument.

        Args:
            chunks: List of chunk dictionaries (from chunker)

        Returns:
            str: Document summary

        Example:
            >>> chunks = chunker.chunk_document(enriched_doc)
            >>> summary = summary_service.generate_summary_from_chunks(chunks)
        """
        logger.info(f"Generating summary from {len(chunks)} chunks")

        try:
            # Combine first few chunks to get document start
            first_chunks = chunks[: min(3, len(chunks))]
            combined_text = "\n\n".join([chunk["content"] for chunk in first_chunks])

            # Extract purpose from combined text
            first_para = self._get_first_paragraph(combined_text)
            if first_para:
                sentences = self._split_into_sentences(first_para)
                purpose = " ".join(sentences[: min(2, len(sentences))])
            else:
                purpose = f"Document with {len(chunks)} sections"

            # Extract key points from all chunks
            key_points = []
            for chunk in chunks[: min(10, len(chunks))]:  # Limit to first 10 chunks
                key_sentences = self._extract_key_sentences(
                    chunk["content"], max_sentences=1
                )
                if key_sentences:
                    key_points.extend(key_sentences)

            # Combine
            summary_parts = [f"Purpose: {purpose}"]
            if key_points:
                summary_parts.append(f"\nKey Points: {' '.join(key_points[:5])}")

            summary = "\n".join(summary_parts)

            # Truncate if needed
            if len(summary) > self.config.max_summary_length:
                summary = summary[: self.config.max_summary_length] + "..."

            logger.info(f"✅ Generated summary from chunks ({len(summary)} chars)")

            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary from chunks: {str(e)}")
            return "Document summary unavailable"


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path

    from src.config import get_config
    from src.services.doc_intel_service import DocumentIntelligenceService

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        config = get_config()
        summary_service = SummaryGenerationService(config.file_summary)

        print("📝 File Summary Generation Service")
        print(f"Method: {config.file_summary.summary_method}")
        print(f"Max sentences per page: {config.file_summary.max_sentences_per_page}")
        print(f"Max summary length: {config.file_summary.max_summary_length}")

        # Test with a file (if provided)
        if len(sys.argv) > 1:
            file_path = Path(sys.argv[1])
            if file_path.exists():
                print(f"\nProcessing: {file_path.name}")

                # Initialize Document Intelligence service
                doc_intel_service = DocumentIntelligenceService(config.doc_intelligence)

                with open(file_path, "rb") as f:
                    # Determine doc type
                    from src.models.document_models import DocumentType

                    ext = file_path.suffix.lower().lstrip(".")
                    doc_type = DocumentType.PDF if ext == "pdf" else DocumentType.DOCX

                    # Analyze document
                    enriched_doc = doc_intel_service.analyze_document(f, file_path.name, doc_type)

                    # Generate summary
                    summary = summary_service.generate_summary(enriched_doc)

                    # Display results
                    print(f"\n✅ Generated Summary ({len(summary)} chars):")
                    print("=" * 80)
                    print(summary)
                    print("=" * 80)

            else:
                print(f"File not found: {file_path}")
                sys.exit(1)
        else:
            print("\nUsage: python summary_generation_service.py <path_to_document>")
            print("\n✅ SummaryGenerationService initialized successfully!")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
