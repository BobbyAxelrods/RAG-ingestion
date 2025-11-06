"""
Compatibility shim for ETL models.

Re-exports all symbols from `src.etl.models.etl_models` so modules that
import `src.models.etl_models` continue to work.
"""

from src.etl.models.etl_models import *  # noqa: F401,F403

