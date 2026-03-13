"""Generate embeddable text chunks from BinomialHash slots.

Each chunk follows the WorkbookIndexer chunk format and includes label,
schema, stats, preview rows, and fingerprint.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def slot_to_chunk(
    key: str,
    label: str,
    fingerprint: str,
    row_count: int,
    rows: List[Dict[str, Any]],
    columns: List[str],
    col_types: Dict[str, str],
    col_stats: Dict[str, Dict[str, Any]],
    *,
    max_columns: int = 30,
    preview_rows: int = 2,
    preview_chars: int = 800,
    max_content_chars: int = 6000,
) -> Dict[str, Any]:
    """Convert a single slot's metadata into an embeddable text chunk."""
    from ..schema import T_DATE, T_NUMERIC, T_STRING

    lines = [f"Dataset: {label} ({row_count} records, {len(columns)} columns)"]
    lines.append(f"BH key: {key} | fingerprint: {fingerprint[:12]}")
    lines.append("Columns:")
    for col in columns[:max_columns]:
        ct = col_types.get(col, "?")
        st = col_stats.get(col, {})
        desc = f"  {col} ({ct})"
        if ct == T_NUMERIC and "min" in st:
            desc += f" range [{st['min']}..{st['max']}] mean={st.get('mean')}"
        elif ct == T_STRING and "top_values" in st:
            desc += f" {st.get('unique_count', '?')} unique, top: {st['top_values'][:3]}"
        elif ct == T_DATE and "min_date" in st:
            desc += f" range [{st['min_date']}..{st['max_date']}]"
        lines.append(desc)

    preview = json.dumps(rows[:preview_rows], default=str)
    if len(preview) > preview_chars:
        preview = preview[:preview_chars] + "...]"
    lines.append(f"Sample: {preview}")

    content = "\n".join(lines)
    if len(content) > max_content_chars:
        content = content[:max_content_chars]

    return {
        "content": content,
        "chunk_type": "bh_dataset",
        "cell_range": None,
        "sheet_name": "_bh",
        "metadata": {
            "bh_key": key,
            "bh_fingerprint": fingerprint[:16],
            "label": label,
            "row_count": row_count,
            "columns": columns[:20],
            "col_types": {c: col_types[c] for c in columns[:20] if c in col_types},
        },
    }
