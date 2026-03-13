"""Exporter helpers for BinomialHash data.

Formats:
- **rows**: clean row dicts for frontend table components
- **csv**: CSV string for file downloads
- **markdown**: GFM table for inline chat rendering
- **excel**: header + values matrix
- **chunks**: embeddable text chunks for vector retrieval
- **artifact**: download-ready wrapper with filename + MIME type
"""

from .artifact import build_artifact
from .chunks import slot_to_chunk
from .csv import export_csv
from .excel import export_excel_batch
from .markdown import export_markdown
from .rows import export_rows

__all__ = [
    "build_artifact",
    "export_csv",
    "export_excel_batch",
    "export_markdown",
    "export_rows",
    "slot_to_chunk",
]
