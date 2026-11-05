"""
app/utils/text_processing.py — HTML/Markdown text-processing helpers.

These functions are used by the upload and WebSocket handlers to normalize
OCR output before saving it to disk and sending it to the frontend.

All functions are pure (no I/O, no side effects) and therefore safe to
unit-test in isolation.
"""

import re
import base64


# ---------------------------------------------------------------------------
# Table markup normalization
# ---------------------------------------------------------------------------

def wrap_naked_tr(text: str) -> str:
    """
    Wrap runs of NAKED <tr>...</tr> (rows the VLM emitted without a <table>)
    in a <table>, WITHOUT touching <tr> that already live inside a well-formed
    <table>.

    A naive regex that wraps every <tr> run corrupts proper tables: it wraps
    the rows inside <thead>/<tbody> in extra nested <table> tags, so ONE table
    then exports as THREE (outer + header-only + data-only).  We first hide
    complete existing <table>...</table> blocks behind placeholders, wrap only
    the naked <tr> that remain, then restore the hidden tables untouched.
    """
    stash: list = []

    def _hide(m: re.Match) -> str:
        stash.append(m.group(0))
        return f"\x00TBL{len(stash) - 1}\x00"

    protected = re.sub(
        r"<table[\s\S]*?</table>", _hide, text, flags=re.IGNORECASE
    )

    def _wrap(m: re.Match) -> str:
        block = m.group(1)
        return (
            '\n<table class="dataframe table table-hover table-striped '
            'table-bordered ocr-dataframe">\n'
            + block.strip()
            + "\n</table>\n"
        )

    protected = re.sub(
        r"((?:[ \t\r\n]*<tr>[\s\S]*?</tr>[ \t\r\n]*)+)",
        _wrap,
        protected,
        flags=re.IGNORECASE,
    )

    for i, tbl in enumerate(stash):
        protected = protected.replace(f"\x00TBL{i}\x00", tbl)

    return protected


def replace_markdown_tables_with_html(text: str) -> str:
    """
    Find markdown tables AND raw HTML tables in *text* and wrap/convert them
    inside a ``<div class="table-container">``.

    Step 1 wraps any existing raw HTML <table>...</table> blocks.
    Step 2 converts markdown pipe-tables to HTML via the ``markdown`` library.
    """
    import markdown

    # 1. Wrap raw HTML tables that the VLM emitted directly.
    text = re.sub(
        r"(<table.*?>.*?</table>)",
        r'<div class="table-container">\n\1\n</div>',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # 2. Detect and convert markdown pipe-tables.
    lines = text.split("\n")
    new_lines: list = []
    table_lines: list = []
    in_table = False

    for line in lines:
        is_table_row = re.match(r"^\s*\|.*\|", line)
        is_separator = in_table and re.match(r"^\s*[-|:\s]+$", line)

        if is_table_row or is_separator:
            in_table = True
            table_lines.append(line)
        else:
            if in_table:
                table_md = "\n".join(table_lines)
                table_html = markdown.markdown(table_md, extensions=["tables"])
                new_lines.append(
                    f'<div class="table-container">\n{table_html}\n</div>'
                )
                table_lines = []
                in_table = False
            new_lines.append(line)

    # Flush any trailing table block.
    if in_table:
        table_md = "\n".join(table_lines)
        table_html = markdown.markdown(table_md, extensions=["tables"])
        new_lines.append(
            f'<div class="table-container">\n{table_html}\n</div>'
        )

    return "\n".join(new_lines)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def smart_truncate_text(text: str, max_chars: int = 3000) -> str:
    """
    Truncate *text* to at most *max_chars*, preferring sentence boundaries.

    Falls back to word boundaries, then hard truncation with "..." appended.
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]

    # Prefer a sentence-ending punctuation mark in the last 20 % of the budget.
    last_boundary = max(
        truncated.rfind("."),
        truncated.rfind("?"),
        truncated.rfind("!"),
    )
    if last_boundary > max_chars * 0.8:
        return truncated[: last_boundary + 1]

    # Fall back to word boundary.
    last_space = truncated.rfind(" ")
    if last_space > 0:
        return truncated[:last_space] + "..."

    return truncated + "..."


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def validate_base64(data: str) -> bool:
    """Return True if *data* is valid base64-encoded content."""
    try:
        base64.b64decode(data, validate=True)
        return True
    except Exception:
        return False
