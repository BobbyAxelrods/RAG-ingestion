"""
Compatibility shim for modules importing from `src.models.document_models`.

Re-exports all symbols from `src.etl.models.document_models`.
"""

from src.etl.models.document_models import *  # noqa: F401,F403

