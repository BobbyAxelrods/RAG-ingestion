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
        # Prefer provided path; if it fails, try artifact Excel fallback; otherwise proceed empty.
        source_path: Path | None = None
        excel_fallback = Path(
            "c:/Users/muhammad.s.safian/OneDrive - Avanade/Project/MAIN_PROJECT/_avanade_main/_avanade_main/_01_prudential/PRUHK-AgenticRAG-V1/artifact/AI-PIL - LifeOps (Cashier, Claims) and GI PIL File List v1.0.xlsx"
        )

        if self.csv_path:
            candidate = Path(self.csv_path)
            if candidate.exists():
                source_path = candidate
        if source_path is None and excel_fallback.exists():
            source_path = excel_fallback
        if source_path is None:
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
                # Helper: read all sheets and concatenate, preserving original headers
                def _read_excel_all_sheets(path: Path, engine: str | None = None) -> pd.DataFrame:
                    kwargs = {"sheet_name": None}
                    if engine:
                        kwargs["engine"] = engine
                    sheets_dict = pd.read_excel(path, **kwargs)
                    dfs: list[pd.DataFrame] = []
                    for sheet_name, df in sheets_dict.items():
                        try:
                            df2 = df.dropna(how="all")
                        except Exception:
                            df2 = df
                        if len(df2.index) == 0 and len(df2.columns) == 0:
                            continue
                        # Tag originating sheet to aid troubleshooting
                        try:
                            df2["__sheet_name__"] = sheet_name
                        except Exception:
                            pass
                        dfs.append(df2)
                    if not dfs:
                        return pd.DataFrame()
                    return pd.concat(dfs, ignore_index=True)

                excel_loaded = False

                # Try primary Excel load (openpyxl for .xlsx)
                try:
                    engine = "openpyxl" if source_path.suffix.lower() == ".xlsx" else None
                    self.metadata_df = _read_excel_all_sheets(source_path, engine)
                    excel_loaded = True
                    try:
                        # Log a few columns for visibility
                        logger.info(
                            f"Loaded Excel with {len(self.metadata_df)} rows; sample columns: "
                            f"{list(map(str, self.metadata_df.columns))[:12]}"
                        )
                    except Exception:
                        pass
                except PermissionError as e_perm:
                    # Workaround OneDrive file lock by copying to a temp file
                    try:
                        tmp_copy = source_path.parent / "_tmp_metadata_copy.xlsx"
                        import shutil
                        shutil.copy2(source_path, tmp_copy)
                        engine = "openpyxl" if tmp_copy.suffix.lower() == ".xlsx" else None
                        self.metadata_df = _read_excel_all_sheets(tmp_copy, engine)
                        excel_loaded = True
                        logger.info(f"Excel was locked; loaded via temp copy: {tmp_copy}")
                    except Exception as e_copy:
                        logger.warning(
                            f"Excel temp-copy load failed ({type(e_copy).__name__}): {e_copy}."
                        )
                except Exception as e_excel_primary:
                    logger.warning(
                        f"Primary Excel load failed ({type(e_excel_primary).__name__}): {e_excel_primary}."
                    )
                    # Try artifact Excel fallback if different
                    if source_path != excel_fallback and excel_fallback.exists():
                        try:
                            logger.info(f"Trying artifact Excel fallback: {excel_fallback}")
                            engine = "openpyxl" if excel_fallback.suffix.lower() == ".xlsx" else None
                            self.metadata_df = _read_excel_all_sheets(excel_fallback, engine)
                            excel_loaded = True
                        except Exception as e_excel_fallback:
                            logger.error(
                                f"Fallback Excel load failed ({type(e_excel_fallback).__name__}): {e_excel_fallback}."
                            )

                # If Excel still not loaded, try a sibling CSV with common encodings
                if not excel_loaded:
                    candidate_csvs: list[Path] = []
                    candidate_csvs.append(source_path.with_suffix(".csv"))
                    candidate_csvs.append(excel_fallback.with_suffix(".csv"))
                    for csv_path in candidate_csvs:
                        if csv_path.exists():
                            logger.info(f"Falling back to CSV at: {csv_path}")
                            for enc in ("utf-8-sig", "utf-8", "latin1"):
                                try:
                                    self.metadata_df = pd.read_csv(csv_path, encoding=enc)
                                    excel_loaded = True
                                    break
                                except Exception:
                                    continue
                            if excel_loaded:
                                break
                    if not excel_loaded:
                        raise RuntimeError("Excel and CSV fallbacks failed")
            else:
                # CSV path: try encodings commonly produced by Excel
                loaded = False
                for enc in ("utf-8"):
                    try:
                        self.metadata_df = pd.read_csv(source_path, encoding=enc)
                        loaded = True
                        break
                    except Exception:
                        continue
                if not loaded:
                    self.metadata_df = pd.read_csv(source_path)

            # Note: Do NOT normalize column names; respect Excel headers as-is

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

            # Ensure filename column exists by mapping commonly used Excel headers, without altering values
            if "filename" not in self.metadata_df.columns:
                # Try common header variants exactly as they may appear in Excel
                raw_cols = list(self.metadata_df.columns)
                filename_source_col: str | None = None
                for candidate in [
                    "filename",
                    "file_name",
                    "File Name",
                    "Filename",
                    "FILE NAME",
                    "FILE_NAME",
                ]:
                    # Match candidate in a case-insensitive way but use the original column name
                    for col in raw_cols:
                        if str(col).lower() == candidate.lower():
                            filename_source_col = col
                            break
                    if filename_source_col:
                        break

                if filename_source_col:
                    # Copy values as-is (no strip/normalization)
                    try:
                        self.metadata_df["filename"] = self.metadata_df[filename_source_col].astype(str)
                        logger.info(f"Using filename column '{filename_source_col}' from metadata source")
                    except Exception as e:
                        logger.warning(f"Failed to map filename column '{filename_source_col}': {e}")
                        self.metadata_df["filename"] = ""
                else:
                    # Attempt to use a title or name column as a proxy
                    for candidate in ["title_name_en", "title_name_tc"]:
                        if candidate in self.metadata_df.columns:
                            self.metadata_df["filename"] = self.metadata_df[candidate].astype(str)
                            logger.info(f"Using proxy column '{candidate}' for filename")
                            break
                    if "filename" not in self.metadata_df.columns:
                        # Create empty column to avoid KeyError downstream
                        self.metadata_df["filename"] = ""

            # Do not strip/normalize filename column; use values exactly as present in Excel

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

        # Search for filename exact match (use Excel values as-is)
        matches = self.metadata_df[self.metadata_df["filename"] == clean_filename]

        # If not found, attempt exact base-name equality without altering Excel values
        if matches.empty:
            try:
                filename_base_series = self.metadata_df["filename"].astype(str).apply(lambda s: Path(s).stem)
                matches = self.metadata_df[filename_base_series == base_name]
            except Exception:
                pass

        # If still not found, try direct equality against common alternative columns without normalization
        if matches.empty:
            try:
                for alt_col in ["file_name", "file_description", "knowledge_name"]:
                    if alt_col in self.metadata_df.columns:
                        col_series = self.metadata_df[alt_col].astype(str)
                        eq_matches = self.metadata_df[col_series == clean_filename]
                        if not eq_matches.empty:
                            matches = eq_matches
                            break
                        # Base-name equality
                        base_eq = self.metadata_df[col_series.apply(lambda s: Path(s).stem) == base_name]
                        if not base_eq.empty:
                            matches = base_eq
                            break
            except Exception:
                pass

        # If not found, try case-insensitive/base-name match and title fallback
        if matches.empty:
            normalized_target = _normalize(clean_filename)
            normalized_base = _normalize(base_name)

            try:
                normalized_col = self.metadata_df["filename"].apply(_normalize)
                matches = self.metadata_df[normalized_col == normalized_target]
                if matches.empty:
                    matches = self.metadata_df[normalized_col == normalized_base]
            except Exception:
                matches = pd.DataFrame()

            # If still empty, try compact normalization (ignoring spaces entirely)
            if matches.empty:
                try:
                    normalized_target_compact = _normalize(clean_filename, compact=True)
                    normalized_base_compact = _normalize(base_name, compact=True)
                    normalized_col_compact = self.metadata_df["filename"].apply(lambda s: _normalize(s, compact=True))
                    matches = self.metadata_df[normalized_col_compact == normalized_target_compact]
                    if matches.empty:
                        matches = self.metadata_df[normalized_col_compact == normalized_base_compact]
                except Exception:
                    pass

            if matches.empty and "title_name_en" in self.metadata_df.columns:
                try:
                    title_col = self.metadata_df["title_name_en"].astype(str).apply(_normalize)
                    # Try equality on normalized forms
                    matches = self.metadata_df[(title_col == normalized_target) | (title_col == normalized_base)]
                    # If still empty, try contains either way to handle minor punctuation/spacing differences
                    if matches.empty:
                        contains_target = title_col.str.contains(normalized_target, regex=False)
                        contains_base = title_col.str.contains(normalized_base, regex=False)
                        matches = self.metadata_df[contains_target | contains_base]
                    # Try compact form comparisons as well
                    if matches.empty:
                        title_col_compact = self.metadata_df["title_name_en"].astype(str).apply(lambda s: _normalize(s, compact=True))
                        target_compact = _normalize(clean_filename, compact=True)
                        base_compact = _normalize(base_name, compact=True)
                        matches = self.metadata_df[(title_col_compact == target_compact) | (title_col_compact == base_compact)]
                        if matches.empty:
                            contains_target_compact = title_col_compact.str.contains(target_compact, regex=False)
                            contains_base_compact = title_col_compact.str.contains(base_compact, regex=False)
                            matches = self.metadata_df[contains_target_compact | contains_base_compact]
                except Exception:
                    pass

            # If still empty, expand lookup to alternative columns by fuzzy/compact matching
            if matches.empty:
                try:
                    target_compact = _normalize(clean_filename, compact=True)
                    base_compact = _normalize(base_name, compact=True)

                    # file_name exact/compact
                    if "file_name" in self.metadata_df.columns:
                        fn = self.metadata_df["file_name"].astype(str)
                        fn_compact = fn.apply(lambda s: _normalize(s, compact=True))
                        matches = self.metadata_df[(fn_compact == target_compact) | (fn_compact == base_compact)]
                        if matches.empty:
                            contains_mask = fn_compact.str.contains(base_compact, regex=False)
                            matches = self.metadata_df[contains_mask]

                    # file_description exact/compact
                    if matches.empty and "file_description" in self.metadata_df.columns:
                        fd = self.metadata_df["file_description"].astype(str)
                        fd_compact = fd.apply(lambda s: _normalize(s, compact=True))
                        matches = self.metadata_df[(fd_compact == target_compact) | (fd_compact == base_compact)]
                        if matches.empty:
                            contains_mask = fd_compact.str.contains(base_compact, regex=False)
                            matches = self.metadata_df[contains_mask]

                    # item_url fuzzy contains: compare filename/base against URL path
                    if matches.empty and "item_url" in self.metadata_df.columns:
                        urls = self.metadata_df["item_url"].astype(str).str.lower()
                        # Use only the last path segment (file-like) for comparison
                        try:
                            url_tail = urls.apply(lambda s: Path(s.split("?")[0]).name if s else s)
                        except Exception:
                            url_tail = urls
                        url_compact = url_tail.apply(lambda s: _normalize(s, compact=True))
                        contains_mask = url_compact.str.contains(base_compact, regex=False) | url_compact.str.contains(target_compact, regex=False)
                        matches = self.metadata_df[contains_mask]

                    # Final fallback: fuzzy similarity across multiple columns using difflib
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
                                # Split on non-alphanumeric, remove empty tokens
                                ta = {t for t in re.split(r"[^A-Za-z0-9]+", a) if t}
                                tb = {t for t in re.split(r"[^A-Za-z0-9]+", b) if t}
                                if not ta or not tb:
                                    return 0.0
                                inter = len(ta & tb)
                                union = len(ta | tb)
                                return inter / union if union else 0.0
                            except Exception:
                                return 0.0

                        candidate_cols = [
                            "filename",
                            "file_name",
                            "file_description",
                            "title_name_en",
                            "item_url",
                        ]

                        best_idx: int | None = None
                        best_score: float = 0.0

                        # Iterate rows and compute max similarity against target/base across candidate columns
                        for idx, row in self.metadata_df.iterrows():
                            scores: list[float] = []
                            for col in candidate_cols:
                                if col in self.metadata_df.columns:
                                    val = str(row.get(col, ""))
                                    if col == "item_url":
                                        try:
                                            val = Path(val.split("?")[0]).name.lower()
                                        except Exception:
                                            val = str(val).lower()
                                    cand = _normalize(val, compact=True)
                                    if cand:
                                        scores.append(_ratio(cand, target_compact))
                                        scores.append(_ratio(cand, base_compact))
                            if scores:
                                score_ratio = max(scores)
                                # Also consider token-level Jaccard similarity
                                score_jaccard = max(
                                    _token_jaccard(cand, target_compact),
                                    _token_jaccard(cand, base_compact),
                                )
                                score = max(score_ratio, score_jaccard)
                                if score > best_score:
                                    best_score = score
                                    best_idx = idx

                        # Use a permissive threshold to accommodate typos/format differences
                        if best_idx is not None and best_score >= 0.72:
                            matches = self.metadata_df.iloc[[best_idx]]
                            try:
                                logger.info(
                                    f"Fuzzy CSV match for '{clean_filename}' with score {best_score:.2f} (row {best_idx})"
                                )
                            except Exception:
                                pass
                except Exception as e:
                    logger.debug(f"Alternative column lookup failed: {str(e)}")

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
