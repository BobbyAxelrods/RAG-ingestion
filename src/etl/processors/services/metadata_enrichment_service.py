"""
CSV Metadata Enrichment Service.

Handles loading CSV metadata and enriching documents with bilingual
metadata (English and Traditional Chinese).
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MetadataConfig
from src.models.document_models import ChunkMetadata, DocumentMetadata

logger = logging.getLogger(__name__)


class MetadataEnrichmentService:
    """
    Service for enriching documents with metadata from CSV.

    Loads CSV at startup and provides lookup functionality for
    enriching documents with bilingual metadata.
    """

    def __init__(self, config: MetadataConfig):
        """
        Initialize Metadata Enrichment service.

        Args:
            config: Metadata configuration

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV is missing required columns
        """
        self.config = config
        self.metadata_df: pd.DataFrame | None = None
        self.csv_path = config.csv_path

        # Required columns for the CSV
        self.required_columns = [
            "filename",
            "document_id",
            "library_name_en",
            "library_name_tc",
            "category_name_en",
            "category_name_tc",
            "title_name_en",
            "title_name_tc",
            "item_url",
        ]

        # Optional columns
        self.optional_columns = ["has_images", "image_count"]

        # Load CSV
        self._load_csv()

        logger.info(
            f"Initialized MetadataEnrichmentService with {len(self.metadata_df)} entries"
        )

    def _load_csv(self):
        """Load and validate CSV file."""
        # Determine source path; support CSV or Excel.
        # Prefer CSV override whenever METADATA_CSV_PATH is set and exists.
        source_path: Path | None = None
        excel_fallback = Path(
            "c:/Users/muhammad.s.safian/OneDrive - Avanade/Project/MAIN_PROJECT/_avanade_main/_avanade_main/_01_prudential/PRUHK-AgenticRAG/indexing_pipeline/AI-PIL - LifeOps (Cashier, Claims) and GI PIL File List v1.0.xlsx"
        )

        if self.csv_path and Path(self.csv_path).exists():
            source_path = self.csv_path
        elif excel_fallback.exists():
            source_path = excel_fallback
        else:
            raise FileNotFoundError(
                f"Metadata file not found. Checked: {self.csv_path} and {excel_fallback}"
            )

        # Log source and detected type
        src_ext = source_path.suffix.lower()
        src_type = "excel" if src_ext in [".xlsx", ".xls"] else "csv"
        logger.info(f"Loading metadata from: {source_path} (type={src_type})")

        try:
            # Attempt to load Excel when extension indicates Excel; otherwise load CSV.
            if src_type == "excel":
                # Try openpyxl for .xlsx, xlrd for .xls where available; fall back gracefully
                try:
                    self.metadata_df = pd.read_excel(source_path, engine="openpyxl")
                except Exception as e_excel_primary:
                    logger.warning(f"Primary Excel load failed ({type(e_excel_primary).__name__}): {e_excel_primary}. Trying alternate engine...")
                    try:
                        # xlrd supports .xls; if not installed or still failing, let pandas choose
                        self.metadata_df = pd.read_excel(source_path)
                    except Exception as e_excel_alt:
                        logger.error(f"Failed to load Excel from {source_path}: {e_excel_alt}. Will attempt CSV fallback if available.")
                        # If Excel fails, attempt CSV fallback path
                        if self.csv_path and Path(self.csv_path).exists():
                            logger.info(f"Falling back to CSV at: {self.csv_path}")
                            # Try common encodings for Excel-exported CSVs
                            loaded = False
                            for enc in ("utf-8-sig", "utf-8", "latin1"):
                                try:
                                    self.metadata_df = pd.read_csv(self.csv_path, encoding=enc)
                                    loaded = True
                                    break
                                except Exception:
                                    continue
                            if not loaded:
                                # Final attempt without explicit encoding
                                self.metadata_df = pd.read_csv(self.csv_path)
                        else:
                            # Re-raise to be caught by outer except
                            raise
            else:
                # CSV path: try encodings commonly produced by Excel
                loaded = False
                for enc in ("utf-8-sig", "utf-8", "latin1"):
                    try:
                        self.metadata_df = pd.read_csv(source_path, encoding=enc)
                        loaded = True
                        break
                    except Exception:
                        continue
                if not loaded:
                    self.metadata_df = pd.read_csv(source_path)

            # Normalize column names: strip, lowercase, replace spaces with underscores
            self.metadata_df.columns = (
                self.metadata_df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
            )

            # Validate required columns
            missing_columns = set(self.required_columns) - set(self.metadata_df.columns)
            if missing_columns:
                # For Excel sources with different schemas, log and proceed; we'll use available columns
                logger.warning(
                    f"Metadata source missing expected columns: {', '.join(missing_columns)}. "
                    f"Proceeding with available columns and storing all extras in additional_fields."
                )

            # Fill NaN values
            self.metadata_df = self.metadata_df.fillna("")

            # Ensure filename column exists; if not, try to construct from known candidates
            if "filename" not in self.metadata_df.columns:
                # Attempt to use a title or name column as a proxy
                for candidate in ["file_name", "name", "title", "title_name_en"]:
                    if candidate in self.metadata_df.columns:
                        self.metadata_df["filename"] = self.metadata_df[candidate].astype(str)
                        break
                if "filename" not in self.metadata_df.columns:
                    # Create empty column to avoid KeyError downstream
                    self.metadata_df["filename"] = ""

            # Strip whitespace from filename column (important for lookup)
            self.metadata_df["filename"] = self.metadata_df["filename"].astype(str).str.strip()

            # Precompute normalized variants to speed matching operations across the DataFrame
            try:
                def _norm_series(series: pd.Series, compact: bool = False) -> pd.Series:
                    import re
                    s = series.astype(str).str.strip().str.lower()
                    s = (
                        s.str.replace(".pdf", "", regex=False)
                         .str.replace(".docx", "", regex=False)
                         .str.replace(".doc", "", regex=False)
                         .str.replace(".xlsx", "", regex=False)
                         .str.replace(".xls", "", regex=False)
                    )
                    s = s.apply(lambda x: re.sub(r"[^a-z0-9]+", " ", x))
                    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
                    if compact:
                        s = s.str.replace(" ", "", regex=False)
                    return s

                # filename normalized forms
                self.metadata_df["filename_norm"] = _norm_series(self.metadata_df["filename"], compact=False)
                self.metadata_df["filename_compact"] = _norm_series(self.metadata_df["filename"], compact=True)

                # Optional columns normalized forms
                if "title_name_en" in self.metadata_df.columns:
                    self.metadata_df["title_name_en_norm"] = _norm_series(self.metadata_df["title_name_en"], compact=False)
                    self.metadata_df["title_name_en_compact"] = _norm_series(self.metadata_df["title_name_en"], compact=True)

                if "file_name" in self.metadata_df.columns:
                    self.metadata_df["file_name_norm"] = _norm_series(self.metadata_df["file_name"], compact=False)
                    self.metadata_df["file_name_compact"] = _norm_series(self.metadata_df["file_name"], compact=True)

                if "file_description" in self.metadata_df.columns:
                    self.metadata_df["file_description_norm"] = _norm_series(self.metadata_df["file_description"], compact=False)
                    self.metadata_df["file_description_compact"] = _norm_series(self.metadata_df["file_description"], compact=True)

                if "item_url" in self.metadata_df.columns:
                    # Extract last path segment then normalize compact for matching
                    urls = self.metadata_df["item_url"].astype(str).str.lower()
                    try:
                        tail = urls.apply(lambda s: Path(s.split("?")[0]).name if s else s)
                    except Exception:
                        tail = urls
                    self.metadata_df["item_url_tail_compact"] = _norm_series(tail, compact=True)
            except Exception as e:
                logger.debug(f"Failed to precompute normalized columns: {e}")

            logger.info(
                f"✅ Loaded {len(self.metadata_df)} metadata entries from metadata source"
            )
            logger.debug(f"Columns: {list(self.metadata_df.columns)}")

        except Exception as e:
            # Report the actual failing source path
            logger.error(f"Failed to load metadata from {source_path}: {str(e)}")
            logger.warning("Proceeding with empty metadata; lookups will use fallback values.")
            # Fallback to empty DataFrame to allow pipeline to proceed
            self.metadata_df = pd.DataFrame(columns=self.required_columns + self.optional_columns)

    def reload_csv(self):
        """
        Reload CSV file.

        Useful if CSV has been updated during runtime.
        """
        logger.info("Reloading metadata CSV...")
        self._load_csv()

    def lookup_metadata(self, filename: str) -> DocumentMetadata:
        """
        Look up metadata for a filename.

        Args:
            filename: Document filename to look up

        Returns:
            DocumentMetadata: Enriched metadata (with fallbacks if not found)

        Example:
            >>> enrichment_service = MetadataEnrichmentService(config)
            >>> metadata = enrichment_service.lookup_metadata("policy.pdf")
            >>> print(metadata.library_name_en)  # "Technical Library"
        """
        # Clean filename for lookup
        clean_filename = filename.strip()
        base_name = Path(clean_filename).stem

        def _normalize(text: str, compact: bool = False) -> str:
            # Lowercase, strip, remove extensions and non-alphanumeric separators
            import re
            t = (
                str(text)
                .strip()
                .lower()
                .replace(".pdf", "")
                .replace(".docx", "")
                .replace(".doc", "")
                .replace(".xlsx", "")
                .replace(".xls", "")
            )
            # Replace any sequence of non-alphanumeric characters with single space
            t = re.sub(r"[^a-z0-9]+", " ", t)
            # Collapse multiple spaces
            t = re.sub(r"\s+", " ", t).strip()
            # Optionally compact by removing all spaces to handle spacing differences (e.g., HealthCheck vs Health Check)
            if compact:
                t = t.replace(" ", "")
            return t

        logger.debug(f"Looking up metadata for: {clean_filename}")

        # Search for filename exact match first
        matches = self.metadata_df[self.metadata_df["filename"] == clean_filename]

        # If not found, use precomputed normalized columns and vectorized masks
        if matches.empty:
            target_norm = _normalize(clean_filename)
            base_norm = _normalize(base_name)
            target_compact = _normalize(clean_filename, compact=True)
            base_compact = _normalize(base_name, compact=True)

            masks: list[pd.Series] = []

            # filename normalized and compact equality
            if "filename_norm" in self.metadata_df.columns:
                masks.append(self.metadata_df["filename_norm"].isin([target_norm, base_norm]))
            if "filename_compact" in self.metadata_df.columns:
                masks.append(self.metadata_df["filename_compact"].isin([target_compact, base_compact]))

            # title_name_en equality/contains (normalized and compact)
            if "title_name_en_norm" in self.metadata_df.columns:
                tnorm = self.metadata_df["title_name_en_norm"]
                masks.append((tnorm == target_norm) | (tnorm == base_norm))
                masks.append(tnorm.str.contains(target_norm, regex=False) | tnorm.str.contains(base_norm, regex=False))
            if "title_name_en_compact" in self.metadata_df.columns:
                tcmp = self.metadata_df["title_name_en_compact"]
                masks.append((tcmp == target_compact) | (tcmp == base_compact))
                masks.append(tcmp.str.contains(target_compact, regex=False) | tcmp.str.contains(base_compact, regex=False))

            # file_name and file_description equality/contains (compact)
            if "file_name_compact" in self.metadata_df.columns:
                fnc = self.metadata_df["file_name_compact"]
                masks.append((fnc == target_compact) | (fnc == base_compact))
                masks.append(fnc.str.contains(base_compact, regex=False))
            if "file_description_compact" in self.metadata_df.columns:
                fdc = self.metadata_df["file_description_compact"]
                masks.append((fdc == target_compact) | (fdc == base_compact))
                masks.append(fdc.str.contains(base_compact, regex=False))

            # item_url tail contains (compact)
            if "item_url_tail_compact" in self.metadata_df.columns:
                urlc = self.metadata_df["item_url_tail_compact"]
                masks.append(urlc.str.contains(target_compact, regex=False) | urlc.str.contains(base_compact, regex=False))

            # Combine masks
            combined = None
            for m in masks:
                combined = m if combined is None else (combined | m)
            if combined is not None:
                result = self.metadata_df[combined]
                if not result.empty:
                    matches = result

            # Fuzzy fallback: compute best score per row using apply across candidate compact columns
            if matches.empty:
                import difflib
                def _ratio(a: str, b: str) -> float:
                    try:
                        return difflib.SequenceMatcher(None, a, b).ratio()
                    except Exception:
                        return 0.0

                def _token_jaccard(a: str, b: str) -> float:
                    try:
                        import re
                        ta = {t for t in re.split(r"[^A-Za-z0-9]+", a) if t}
                        tb = {t for t in re.split(r"[^A-Za-z0-9]+", b) if t}
                        if not ta or not tb:
                            return 0.0
                        inter = len(ta & tb)
                        union = len(ta | tb)
                        return inter / union if union else 0.0
                    except Exception:
                        return 0.0

                cand_cols = [
                    "filename_compact",
                    "file_name_compact",
                    "file_description_compact",
                    "title_name_en_compact",
                    "item_url_tail_compact",
                ]

                def _row_best_score(row: pd.Series) -> float:
                    best = 0.0
                    for col in cand_cols:
                        if col in row and isinstance(row[col], str):
                            cand = row[col]
                            if cand:
                                sr = max(_ratio(cand, target_compact), _ratio(cand, base_compact))
                                sj = max(_token_jaccard(cand, target_compact), _token_jaccard(cand, base_compact))
                                best = max(best, sr, sj)
                    return best

                try:
                    scores = self.metadata_df.apply(_row_best_score, axis=1)
                    best_score = float(scores.max()) if len(scores) > 0 else 0.0
                    if best_score >= 0.72:
                        best_idx = int(scores.idxmax())
                        matches = self.metadata_df.iloc[[best_idx]]
                        logger.info(
                            f"Fuzzy CSV match for '{clean_filename}' with score {best_score:.2f} (row {best_idx})"
                        )
                except Exception as e:
                    logger.debug(f"Fuzzy vectorized scoring failed: {e}")

        if matches.empty:
            logger.warning(
                f"Filename '{clean_filename}' not found in CSV, using fallback metadata"
            )
            return self._create_fallback_metadata(clean_filename)

        # Get first match (there should only be one)
        row = matches.iloc[0]

        # Extract all fields (safe access in case some columns are missing)
        document_id_val = row.get("document_id", None)
        metadata = DocumentMetadata(
            file_name=clean_filename,
            document_id=str(document_id_val) if document_id_val not in ["", None] else None,
            library_name_en=str(row.get("library_name_en", "Unknown")) or "Unknown",
            library_name_tc=str(row.get("library_name_tc", "未知")) or "未知",
            category_name_en=str(row.get("category_name_en", "Uncategorized")) or "Uncategorized",
            category_name_tc=str(row.get("category_name_tc", "未分類")) or "未分類",
            title_name_en=str(row.get("title_name_en", clean_filename)) or clean_filename,
            title_name_tc=str(row.get("title_name_tc", clean_filename)) or clean_filename,
            item_url=str(row.get("item_url", "")) or "",
            has_images=bool(row.get("has_images", False)),
            image_count=int(row.get("image_count", 0)),
        )

        # Add any additional CSV columns to additional_fields
        additional_cols = set(self.metadata_df.columns) - set(self.required_columns) - set(
            self.optional_columns
        )
        for col in additional_cols:
            # Use safe get to avoid KeyError
            try:
                metadata.additional_fields[col] = str(row.get(col, ""))
            except Exception:
                metadata.additional_fields[col] = ""

        logger.debug(
            f"Found metadata for '{clean_filename}': doc_id={metadata.document_id}, "
            f"library={metadata.library_name_en}"
        )

        return metadata

    def _create_fallback_metadata(self, filename: str) -> DocumentMetadata:
        """
        Create fallback metadata for files not in CSV.

        Args:
            filename: Document filename

        Returns:
            DocumentMetadata: Fallback metadata with "Unknown" values
        """
        # Extract title from filename
        from pathlib import Path

        title = Path(filename).stem.replace("_", " ").replace("-", " ").title()

        return DocumentMetadata(
            file_name=filename,
            document_id=None,
            library_name_en="Unknown",
            library_name_tc="未知",
            category_name_en="Uncategorized",
            category_name_tc="未分類",
            title_name_en=title,
            title_name_tc=title,
            item_url="",
            has_images=False,
            image_count=0,
        )

    def enrich_chunk_metadata(
        self,
        filename: str,
        page_number: int | None = None,
        chunk_section: str = "",
        has_images: bool = False,
        image_count: int = 0,
        image_types: list[str] | None = None,
        image_page_numbers: list[int] | None = None,
    ) -> ChunkMetadata:
        """
        Create enriched chunk metadata.

        Combines chunk-specific info with CSV metadata.

        Args:
            filename: Document filename
            page_number: Page number of chunk
            chunk_section: Section identifier
            has_images: Whether chunk contains image descriptions
            image_count: Number of images in chunk
            image_types: Types of images (chart, diagram, etc.)
            image_page_numbers: Page numbers where images appear

        Returns:
            ChunkMetadata: Enriched chunk metadata

        Example:
            >>> chunk_meta = enrichment_service.enrich_chunk_metadata(
            >>>     filename="policy.pdf",
            >>>     page_number=5,
            >>>     chunk_section="Premium Calculation",
            >>>     has_images=True,
            >>>     image_count=2
            >>> )
        """
        # Look up document metadata from CSV
        doc_metadata = self.lookup_metadata(filename)

        # Create chunk metadata with CSV enrichment
        chunk_metadata = ChunkMetadata(
            page_number=page_number,
            chunk_section=chunk_section,
            has_images=has_images,
            image_count=image_count,
            image_types=image_types or [],
            image_page_numbers=image_page_numbers or [],
            # Add CSV fields
            document_id=doc_metadata.document_id,
            library_name_en=doc_metadata.library_name_en,
            library_name_tc=doc_metadata.library_name_tc,
            category_name_en=doc_metadata.category_name_en,
            category_name_tc=doc_metadata.category_name_tc,
            # Add any additional CSV fields
            additional_fields=doc_metadata.additional_fields,
        )

        return chunk_metadata

    def get_all_filenames(self) -> list[str]:
        """
        Get list of all filenames in the CSV.

        Returns:
            list[str]: List of filenames

        Example:
            >>> filenames = enrichment_service.get_all_filenames()
            >>> print(f"CSV contains {len(filenames)} files")
        """
        if self.metadata_df is None:
            return []
        return self.metadata_df["filename"].tolist()

    def filename_exists(self, filename: str) -> bool:
        """
        Check if filename exists in CSV.

        Args:
            filename: Filename to check

        Returns:
            bool: True if filename found in CSV
        """
        clean_filename = filename.strip()
        return clean_filename in self.metadata_df["filename"].values

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the metadata CSV.

        Returns:
            dict: Statistics including counts by library, category, etc.

        Example:
            >>> stats = enrichment_service.get_statistics()
            >>> print(f"Total files: {stats['total_files']}")
        """
        if self.metadata_df is None:
            return {}

        stats = {
            "total_files": len(self.metadata_df),
            "unique_libraries": self.metadata_df["library_name_en"].nunique(),
            "unique_categories": self.metadata_df["category_name_en"].nunique(),
            "files_with_images": self.metadata_df.get("has_images", pd.Series([False])).sum(),
            "total_images": self.metadata_df.get("image_count", pd.Series([0])).sum(),
            "libraries": self.metadata_df["library_name_en"].value_counts().to_dict(),
            "categories": self.metadata_df["category_name_en"].value_counts().to_dict(),
        }

        return stats

    def validate_csv(self) -> tuple[bool, list[str]]:
        """
        Validate CSV for common issues.

        Returns:
            tuple: (is_valid, list of issues)

        Example:
            >>> is_valid, issues = enrichment_service.validate_csv()
            >>> if not is_valid:
            >>>     for issue in issues:
            >>>         print(f"⚠️ {issue}")
        """
        issues = []

        if self.metadata_df is None:
            issues.append("CSV not loaded")
            return False, issues

        # Check for duplicate filenames
        duplicate_filenames = self.metadata_df[
            self.metadata_df.duplicated(subset=["filename"], keep=False)
        ]["filename"].tolist()
        if duplicate_filenames:
            issues.append(f"Duplicate filenames found: {', '.join(set(duplicate_filenames))}")

        # Check for missing document IDs
        missing_ids = self.metadata_df[self.metadata_df["document_id"].isna()]
        if not missing_ids.empty:
            issues.append(
                f"{len(missing_ids)} entries missing document_id: "
                f"{', '.join(missing_ids['filename'].tolist()[:5])}"
            )

        # Check for empty required fields
        for col in ["library_name_en", "category_name_en", "title_name_en"]:
            empty = self.metadata_df[self.metadata_df[col].str.strip() == ""]
            if not empty.empty:
                issues.append(
                    f"{len(empty)} entries with empty {col}: "
                    f"{', '.join(empty['filename'].tolist()[:5])}"
                )

        # Check for Chinese characters in TC fields
        if "library_name_tc" in self.metadata_df.columns:
            non_chinese = self.metadata_df[
                ~self.metadata_df["library_name_tc"].str.contains(r"[\u4e00-\u9fff]", na=False)
            ]
            if not non_chinese.empty and len(non_chinese) > len(self.metadata_df) * 0.5:
                issues.append(
                    f"Warning: {len(non_chinese)} entries may be missing Traditional Chinese translations"
                )

        is_valid = len(issues) == 0
        return is_valid, issues


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
        enrichment_service = MetadataEnrichmentService(config.metadata)

        print("📊 Metadata CSV Statistics\n")

        # Get statistics
        stats = enrichment_service.get_statistics()
        print(f"Total files: {stats['total_files']}")
        print(f"Unique libraries: {stats['unique_libraries']}")
        print(f"Unique categories: {stats['unique_categories']}")
        print(f"Files with images: {stats['files_with_images']}")
        print(f"Total images: {stats['total_images']}")

        print("\n📚 Libraries:")
        for library, count in stats["libraries"].items():
            print(f"  - {library}: {count} files")

        print("\n📁 Categories:")
        for category, count in stats["categories"].items():
            print(f"  - {category}: {count} files")

        # Validate CSV
        print("\n✅ Validating CSV...")
        is_valid, issues = enrichment_service.validate_csv()
        if is_valid:
            print("✅ CSV is valid!")
        else:
            print("⚠️ CSV has issues:")
            for issue in issues:
                print(f"  - {issue}")

        # Test lookup
        print("\n🔍 Testing metadata lookup...")
        filenames = enrichment_service.get_all_filenames()
        if filenames:
            test_filename = filenames[0]
            print(f"\nLooking up: {test_filename}")
            metadata = enrichment_service.lookup_metadata(test_filename)
            print(f"  Document ID: {metadata.document_id}")
            print(f"  Library (EN): {metadata.library_name_en}")
            print(f"  Library (TC): {metadata.library_name_tc}")
            print(f"  Category (EN): {metadata.category_name_en}")
            print(f"  Category (TC): {metadata.category_name_tc}")
            print(f"  Title (EN): {metadata.title_name_en}")
            print(f"  Title (TC): {metadata.title_name_tc}")
            print(f"  URL: {metadata.item_url}")

        # Test missing file
        print("\n🔍 Testing missing file fallback...")
        missing_metadata = enrichment_service.lookup_metadata("nonexistent_file.pdf")
        print(f"  Library (EN): {missing_metadata.library_name_en}")
        print(f"  Category (EN): {missing_metadata.category_name_en}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
