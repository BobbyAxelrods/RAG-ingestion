"""
Centralized configuration management for the Document Ingestion Pipeline.

This module loads and validates all configuration from environment variables
using Pydantic Settings for type safety and validation.
"""

from pathlib import Path
from typing import Literal
import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureBlobConfig(BaseSettings):
    """Azure Blob Storage configuration."""

    # Make optional to support local-only runs
    connection_string: str = Field(default="", alias="AZURE_STORAGE_CONNECTION_STRING")
    container_name: str = Field(default="", alias="STORAGE_CONTAINER_NAME")

    # Disable .env parsing for this section to avoid strict list parsing errors
    # Defaults are sensible; override via OS env if needed.
    model_config = SettingsConfigDict(env_file=None, extra="ignore")


class DocumentIntelligenceConfig(BaseSettings):
    """Azure Document Intelligence configuration."""

    endpoint: str = Field(..., alias="DOC_INTELLIGENCE_ENDPOINT")
    key: str = Field(..., alias="DOC_INTELLIGENCE_KEY")
    api_version: str = Field(default="2024-07-31-preview", alias="DOC_INTELLIGENCE_API_VERSION")
    # Timeout and retry controls
    timeout_seconds: int = Field(default=300, alias="DOC_INTELLIGENCE_TIMEOUT_SECONDS")
    retry_attempts: int = Field(default=2, alias="DOC_INTELLIGENCE_RETRY_ATTEMPTS")
    retry_delay_seconds: int = Field(default=5, alias="DOC_INTELLIGENCE_RETRY_DELAY_SECONDS")

    # Load from .env so local runs can use provided credentials
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class AzureSearchConfig(BaseSettings):
    """Azure AI Search configuration."""

    # Primary fields
    endpoint: str = Field(default="", alias="SEARCH_SERVICE_ENDPOINT")
    key: str = Field(default="", alias="SEARCH_SERVICE_KEY")
    index_name: str = Field(default="", alias="SEARCH_INDEX_NAME")
    # Optional separate index name for ETL hierarchical schema
    etl_index_name: str = Field(default="", alias="ETL_SEARCH_INDEX_NAME")

    # Alternate env keys (will be consolidated via model validator)
    endpoint_alt1: str = Field(default="", alias="AZURE_SEARCH_SERVICE_ENDPOINT")
    endpoint_alt2: str = Field(default="", alias="WEBSEARCH_AI_INDEX_ENDPOINT")
    endpoint_alt3: str = Field(default="", alias="CSV_AI_INDEX_ENDPOINT")

    key_alt1: str = Field(default="", alias="AZURE_SEARCH_API_KEY")
    key_alt2: str = Field(default="", alias="WEBSEARCH_AI_INDEX_API_KEY")
    key_alt3: str = Field(default="", alias="CSV_AI_INDEX_API_KEY")

    index_alt1: str = Field(default="", alias="AZURE_SEARCH_INDEX")
    index_alt2: str = Field(default="", alias="AISEARCH_INDEX_NAME")
    index_alt3: str = Field(default="", alias="WEBSEARCH_AI_INDEX_NAME")
    index_alt4: str = Field(default="", alias="CSV_AI_INDEX_NAME")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("endpoint", mode="after")
    def _fill_endpoint(cls, v, info):
        if v:
            return v
        # Try alternates
        for alt in [
            info.data.get("endpoint_alt1"),
            info.data.get("endpoint_alt2"),
            info.data.get("endpoint_alt3"),
        ]:
            if alt:
                return alt
        return v

    @field_validator("key", mode="after")
    def _fill_key(cls, v, info):
        if v:
            return v
        for alt in [
            info.data.get("key_alt1"),
            info.data.get("key_alt2"),
            info.data.get("key_alt3"),
        ]:
            if alt:
                return alt
        return v

    @field_validator("index_name", mode="after")
    def _fill_index(cls, v, info):
        if v:
            return v
        for alt in [
            info.data.get("index_alt1"),
            info.data.get("index_alt2"),
            info.data.get("index_alt3"),
            info.data.get("index_alt4"),
        ]:
            if alt:
                return alt
        return v

    @field_validator("etl_index_name", mode="after")
    def _fill_etl_index(cls, v, info):
        # Default to primary index_name if ETL index not provided
        if v:
            return v
        primary = info.data.get("index_name")
        return primary


class AzureOpenAIConfig(BaseSettings):
    """Azure OpenAI configuration for embeddings."""

    # Primary fields (embeddings)
    endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    key: str = Field(default="", alias="AZURE_OPENAI_KEY")
    deployment_name: str = Field(default="", alias="AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version: str = Field(default="2024-12-01-preview", alias="AZURE_OPENAI_API_VERSION")
    embedding_dimensions: int = Field(default=1536, alias="EMBEDDING_DIMENSIONS")

    # Alternate env keys
    embedding_endpoint: str = Field(default="", alias="EMBEDDING_ENDPOINT")
    chat_endpoint: str = Field(default="", alias="CHAT_MODEL_ENDPOINT")
    foundry_endpoint: str = Field(default="", alias="FOUNDRY_PROJECT_ENDPOINT")
    ai_project_endpoint: str = Field(default="", alias="AIPROJECT_ENDPOINT")

    embedding_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    chat_key: str = Field(default="", alias="CHAT_MODEL_API_KEY")
    generic_api_key: str = Field(default="", alias="API_KEY")

    deployment_alt1: str = Field(default="", alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    deployment_alt2: str = Field(default="", alias="EMBEDDING_MODEL")
    deployment_alt3: str = Field(default="", alias="FOUNDRY_MODEL_DEPLOYMENT_NAME")
    deployment_alt4: str = Field(default="", alias="CHAT_MODEL")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("endpoint", mode="after")
    def _fill_endpoint(cls, v, info):
        if v:
            return v
        for alt in [
            info.data.get("embedding_endpoint"),
            info.data.get("chat_endpoint"),
            info.data.get("foundry_endpoint"),
            info.data.get("ai_project_endpoint"),
        ]:
            if alt:
                return alt
        return v

    @field_validator("key", mode="after")
    def _fill_key(cls, v, info):
        if v:
            return v
        for alt in [
            info.data.get("embedding_key"),
            info.data.get("chat_key"),
            info.data.get("generic_api_key"),
        ]:
            if alt:
                return alt
        return v

    @field_validator("deployment_name", mode="after")
    def _fill_deployment(cls, v, info):
        if v:
            return v
        for alt in [
            info.data.get("deployment_alt1"),
            info.data.get("deployment_alt2"),
            info.data.get("deployment_alt3"),
            info.data.get("deployment_alt4"),
        ]:
            if alt:
                return alt
        return v


class AzureOpenAIVisionConfig(BaseSettings):
    """Azure OpenAI GPT-4 Vision configuration for image descriptions."""

    endpoint: str = Field(default="", alias="AZURE_OPENAI_VISION_ENDPOINT")
    key: str = Field(default="", alias="AZURE_OPENAI_VISION_KEY")
    deployment: str = Field(default="gpt-4-vision-preview", alias="AZURE_OPENAI_VISION_DEPLOYMENT")
    api_version: str = Field(
        default="2024-12-01-preview", alias="AZURE_OPENAI_VISION_API_VERSION"
    )

    # Alternate env keys
    endpoint_alt1: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    endpoint_alt2: str = Field(default="", alias="CHAT_MODEL_ENDPOINT")
    key_alt1: str = Field(default="", alias="AZURE_OPENAI_API_KEY")
    key_alt2: str = Field(default="", alias="CHAT_MODEL_API_KEY")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("endpoint", mode="after")
    def _fill_endpoint(cls, v, info):
        if v:
            return v
        for alt in [info.data.get("endpoint_alt1"), info.data.get("endpoint_alt2")]:
            if alt:
                return alt
        return v

    @field_validator("key", mode="after")
    def _fill_key(cls, v, info):
        if v:
            return v
        for alt in [info.data.get("key_alt1"), info.data.get("key_alt2")]:
            if alt:
                return alt
        return v


class MetadataConfig(BaseSettings):
    """Metadata CSV configuration."""

    csv_path: Path = Field(default=Path("./metadata/document_metadata.csv"), alias="METADATA_CSV_PATH")

    @field_validator("csv_path", mode="before")
    @classmethod
    def validate_csv_path(cls, v):
        """Convert string to Path if needed."""
        if isinstance(v, str):
            return Path(v)
        return v

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class ProcessingConfig(BaseSettings):
    """Document processing configuration."""

    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    max_concurrent_processing: int = Field(default=5, alias="MAX_CONCURRENT_PROCESSING")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Backwards compatibility alias for existing chunker imports
class ChunkingConfig(ProcessingConfig):
    """Alias maintained for compatibility with chunker modules.

    Historically the pipeline referenced `ChunkingConfig`. The processing
    settings were consolidated under `ProcessingConfig`, but several modules
    still import `ChunkingConfig`. This subclass preserves that import while
    sharing the same fields and environment bindings as `ProcessingConfig`.
    """
    pass


class FileSummaryConfig(BaseSettings):
    """File summary generation configuration."""

    generate_summary: bool = Field(default=True, alias="GENERATE_FILE_SUMMARY")
    # Align with SummaryGenerationService expected names
    summary_method: Literal["extractive", "llm", "hybrid"] = Field(default="extractive", alias="FILE_SUMMARY_METHOD")
    max_summary_length: int = Field(default=2000, alias="SUMMARY_MAX_LENGTH")
    max_sentences_per_page: int = Field(default=2, alias="SUMMARY_SENTENCES_PER_PAGE")
    include_page_summaries: bool = Field(default=True, alias="SUMMARY_INCLUDE_PAGE_SUMMARIES")
    # Keep legacy fields for backward compatibility (not used by current service)
    method: Literal["extractive", "llm", "hybrid"] = Field(default="extractive", alias="FILE_SUMMARY_METHOD")
    max_length: int = Field(default=2000, alias="SUMMARY_MAX_LENGTH")
    sentences_per_page: int = Field(default=2, alias="SUMMARY_SENTENCES_PER_PAGE")
    include_purpose: bool = Field(default=True, alias="SUMMARY_INCLUDE_PURPOSE")
    include_page_numbers: bool = Field(default=True, alias="SUMMARY_INCLUDE_PAGE_NUMBERS")
    sample_large_docs: bool = Field(default=True, alias="SUMMARY_SAMPLE_LARGE_DOCS")
    large_doc_threshold: int = Field(default=100, alias="SUMMARY_LARGE_DOC_THRESHOLD")

    # Key terms for sentence scoring (comma-separated in .env)
    key_terms: list[str] = Field(default_factory=lambda: [
        "policy", "premium", "coverage", "claim", "benefit",
        "requirement", "eligibility", "applicant", "insured",
        "保單", "保費", "保障", "索償", "要求"
    ], alias="SUMMARY_KEY_TERMS")

    action_words: list[str] = Field(default_factory=lambda: [
        "must", "shall", "should", "required", "mandatory",
        "provides", "includes", "必須", "應該", "需要"
    ], alias="SUMMARY_ACTION_WORDS")

    @field_validator("key_terms", "action_words", mode="before")
    @classmethod
    def parse_comma_separated(cls, v):
        """Robustly parse env values into a list of strings.

        Supports:
        - Comma-separated strings: "a, b, c"
        - JSON-style lists: "[\"a\", \"b\"]"
        - Already-parsed lists
        Falls back to empty list if parsing is not possible.
        """
        if v is None:
            return []
        if isinstance(v, str):
            import json
            s = v.strip()
            # Try JSON parsing first for brackets
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("\"") and s.endswith("\"")):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        return [str(term).strip() for term in parsed if str(term).strip()]
                except Exception:
                    pass
            # Fallback to comma split
            return [term.strip() for term in s.split(",") if term.strip()]
        if isinstance(v, (list, tuple)):
            return [str(term).strip() for term in v if str(term).strip()]
        # Fallback: coerce to string then split
        try:
            return [term.strip() for term in str(v).split(",") if term.strip()]
        except Exception:
            return []

    # Disable .env parsing to avoid DotEnv list parsing issues for key_terms
    model_config = SettingsConfigDict(env_file=None, extra="ignore")


class ImageProcessingConfig(BaseSettings):
    """Image processing configuration."""

    extract_embedded_images: bool = Field(default=True, alias="EXTRACT_EMBEDDED_IMAGES")
    ocr_enabled: bool = Field(default=True, alias="IMAGE_OCR_ENABLED")
    summary_method: Literal["llm", "extractive", "disabled"] = Field(
        default="llm", alias="IMAGE_SUMMARY_METHOD"
    )
    summary_max_length: int = Field(default=200, alias="IMAGE_SUMMARY_MAX_LENGTH")
    include_context_in_chunks: bool = Field(
        default=True, alias="INCLUDE_IMAGE_CONTEXT_IN_CHUNKS"
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: str = Field(default="INFO", alias="LOG_LEVEL")
    file: Path = Field(default=Path("./logs/ingestion_pipeline.log"), alias="LOG_FILE")
    format: Literal["json", "text"] = Field(default="json", alias="LOG_FORMAT")

    @field_validator("file", mode="before")
    @classmethod
    def validate_log_file(cls, v):
        """Convert string to Path and ensure directory exists."""
        if isinstance(v, str):
            v = Path(v)
        v.parent.mkdir(parents=True, exist_ok=True)
        return v

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class PerformanceConfig(BaseSettings):
    """Performance and retry configuration."""

    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    retry_backoff_factor: int = Field(default=2, alias="RETRY_BACKOFF_FACTOR")
    request_timeout: int = Field(default=120, alias="REQUEST_TIMEOUT")
    batch_size_embeddings: int = Field(default=16, alias="BATCH_SIZE_EMBEDDINGS")
    batch_size_search_upload: int = Field(default=1000, alias="BATCH_SIZE_SEARCH_UPLOAD")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class AuthConfig(BaseSettings):
    """Authentication configuration."""

    method: Literal["api_key", "managed_identity"] = Field(default="api_key", alias="AUTH_METHOD")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class RuntimeConfig(BaseSettings):
    """Runtime controls for local/offline runs.

    When `offline_mode` is enabled, the pipeline will avoid initializing
    external services (Azure OpenAI, Azure AI Search, Document Intelligence)
    and use local fallbacks so commands can run end-to-end.
    """

    offline_mode: bool = Field(default=True, alias="OFFLINE_MODE")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class Config:
    """
    Centralized configuration for the entire ingestion pipeline.

    All configuration is loaded from environment variables (.env file).
    Pydantic validates types and provides defaults.

    Usage:
        from src.config import Config

        config = Config()
        print(config.azure_blob.container_name)
        print(config.processing.chunk_size)
    """

    def __init__(self):
        """Initialize all configuration sections."""
        self.azure_blob = AzureBlobConfig()
        self.doc_intelligence = DocumentIntelligenceConfig()
        self.azure_search = AzureSearchConfig()
        self.azure_openai = AzureOpenAIConfig()
        self.azure_openai_vision = AzureOpenAIVisionConfig()
        self.metadata = MetadataConfig()
        self.processing = ProcessingConfig()
        # Provide `chunking` for modules still expecting this section
        self.chunking = ChunkingConfig()
        self.file_summary = FileSummaryConfig()
        self.image_processing = ImageProcessingConfig()
        self.logging = LoggingConfig()
        self.performance = PerformanceConfig()
        self.auth = AuthConfig()
        self.runtime = RuntimeConfig()

    def validate(self) -> bool:
        """
        Validate that all required configuration is present and correct.

        Returns:
            True if all configuration is valid, raises exception otherwise
        """
        # In offline mode, skip strict assertions to allow local runs
        if not self.runtime.offline_mode:
            # Check critical endpoints
            assert self.doc_intelligence.endpoint, "Document Intelligence endpoint is required"
            assert self.azure_search.endpoint, "Azure Search endpoint is required"
            assert self.azure_openai.endpoint, "Azure OpenAI endpoint is required"

            # Check critical keys (unless using managed identity)
            if self.auth.method == "api_key":
                assert self.doc_intelligence.key, "Document Intelligence key is required"
                assert self.azure_search.key, "Azure Search key is required"
                assert self.azure_openai.key, "Azure OpenAI key is required"
        else:
            print("Running in OFFLINE_MODE: external service validations are skipped.")

        # Check metadata CSV exists
        if not self.metadata.csv_path.exists():
            print(f"Warning: Metadata CSV not found at {self.metadata.csv_path}")

        return True

    def __repr__(self) -> str:
        """Return string representation (safe, doesn't expose keys)."""
        return (
            f"Config(\n"
            f"  blob_container={self.azure_blob.container_name},\n"
            f"  search_index={self.azure_search.index_name},\n"
            f"  chunk_size={self.processing.chunk_size},\n"
            f"  auth_method={self.auth.method}\n"
            f")"
        )


# Global config instance (lazy loaded)
_config_instance: Config | None = None


def get_config() -> Config:
    """
    Get the global configuration instance (singleton pattern).

    Returns:
        Config: The configuration instance

    Usage:
        from src.config import get_config

        config = get_config()
        container = config.azure_blob.container_name
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
        _config_instance.validate()
    return _config_instance
