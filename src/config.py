"""
Compatibility shim for modules importing from `src.config`.

Re-exports configuration classes and helpers from `src.etl.config` to
avoid changing existing import paths across the codebase.
"""

from src.etl.config import (
    Config,
    get_config,
    AzureBlobConfig,
    AzureSearchConfig,
    AzureOpenAIConfig,
    AzureOpenAIVisionConfig,
    DocumentIntelligenceConfig,
    MetadataConfig,
    ProcessingConfig,
    ChunkingConfig,
    FileSummaryConfig,
    ImageProcessingConfig,
    LoggingConfig,
    PerformanceConfig,
    AuthConfig,
    RuntimeConfig,
)

__all__ = [
    "Config",
    "get_config",
    "AzureBlobConfig",
    "AzureSearchConfig",
    "AzureOpenAIConfig",
    "AzureOpenAIVisionConfig",
    "DocumentIntelligenceConfig",
    "MetadataConfig",
    "ProcessingConfig",
    "ChunkingConfig",
    "FileSummaryConfig",
    "ImageProcessingConfig",
    "LoggingConfig",
    "PerformanceConfig",
    "AuthConfig",
    "RuntimeConfig",
]

