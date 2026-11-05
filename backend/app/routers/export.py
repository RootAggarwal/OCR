"""
app/routers/export.py — Page editing and Excel export endpoints.

Endpoints
---------
POST /update-page/{session_id}/{page_num}       — overwrite saved OCR text for a page
GET  /export-table/{session_id}/{page_num}       — export one page's tables to .xlsx
GET  /export-all/{session_id}                   — export ALL pages' tables to one .xlsx
"""

import glob
import os
from datetime import datetime
from typing import Optional

import openpyxl
from fastapi import APIRouter
from fastapi.responses import FileResponse
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet
from pydantic import BaseModel

from app.config import TEMP_DIR
from app.parsers.table_parser import TableParser
from app.utils.responses import create_error_response, is_safe_session_id

router = APIRouter(tags=["Export"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class UpdatePageRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Shared Excel style constants.
_HEADER_FILL = PatternFill("solid", fgColor="D9E1F2")
_HEADER_FONT = Font(bold=True, name="Arial")
_NORMAL_FONT = Font(name="Arial")
_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _write_excel_sheet(ws: Worksheet, table_data: list) -> None:
    """
    Write *table_data* (a grid from TableParser) onto *ws*.

    Handles header styling, merged cells (colspan/rowspan), and auto-fit
    column widths.  Extracted to eliminate the copy-paste between
    export_table and export_all_tables.
    """
    for row_idx, row in enumerate(table_data, start=1):
        for col_idx, cell in enumerate(row, start=1):
            if not cell.get("is_top_left", True):
                continue  # span-ghost cell — already merged
            ws_cell = ws.cell(row=row_idx, column=col_idx, value=cell["text"])
            ws_cell.font = _HEADER_FONT if cell.get("header") else _NORMAL_FONT
            ws_cell.alignment = _CENTER
            if cell.get("header"):
                ws_cell.fill = _HEADER_FILL

            colspan = cell.get("colspan", 1)
            rowspan = cell.get("rowspan", 1)
            if colspan > 1 or rowspan > 1:
                ws.merge_cells(
                    start_row=row_idx,
                    start_column=col_idx,
                    end_row=row_idx + rowspan - 1,
                    end_column=col_idx + colspan - 1,
                )

    # Auto-fit column widths (capped at 50 characters).
    for col_idx_num, col in enumerate(ws.columns, start=1):
        max_len = max(
            (len(str(c.value or "")) for c in col if hasattr(c, "value")),
            default=10,
        )
        ws.column_dimensions[get_column_letter(col_idx_num)].width = min(
            max_len + 4, 50
        )


def _page_num_from_path(filepath: str) -> int:
    """Extract the page number from a temp file name like ``temp_{sid}_page_{N}.txt``."""
    try:
        return int(os.path.basename(filepath).split("_")[-1].replace(".txt", ""))
    except Exception:
        return 9999


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/update-page/{session_id}/{page_num}")
async def update_page_text(
    session_id: str, page_num: int, req: UpdatePageRequest
):
    """Overwrite the saved OCR text/HTML for a specific page."""
    try:
        if not is_safe_session_id(session_id):
            return create_error_response("Invalid session_id", status_code=400)
        file_path = os.path.join(TEMP_DIR, f"temp_{session_id}_page_{page_num}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(req.text)
        return {"status": "success", "message": "Page updated successfully."}
    except Exception as e:
        print(f"[ERROR] Failed to update page text: {e}")
        return create_error_response(str(e))


@router.get("/export-table/{session_id}/{page_num}")
async def export_table(
    session_id: str, page_num: int, table_index: Optional[int] = None
):
    """Export HTML tables from one page to an Excel file."""
    try:
        if not is_safe_session_id(session_id):
            return create_error_response("Invalid session_id", status_code=400)

        input_file = os.path.join(
            TEMP_DIR, f"temp_{session_id}_page_{page_num}.txt"
        )
        if not os.path.exists(input_file):
            return create_error_response(
                f"No data found for page {page_num}. "
                f"Make sure the document has been processed first (session: {session_id}).",
                status_code=404,
            )

        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        parser = TableParser()
        parser.feed(content)

        if not parser.tables:
            return create_error_response(
                "No HTML tables found in the page.", status_code=404
            )

        if table_index is not None:
            if table_index < 0 or table_index >= len(parser.tables):
                return create_error_response(
                    f"Invalid table index {table_index}. "
                    f"Page only has {len(parser.tables)} table(s).",
                    status_code=400,
                )
            tables_to_export = [parser.tables[table_index]]
            output_filename = f"page_{page_num}_table_{table_index + 1}.xlsx"
        else:
            tables_to_export = parser.tables
            output_filename = f"page_{page_num}_tables.xlsx"

        output_path = os.path.join(
            TEMP_DIR, f"temp_{session_id}_page_{page_num}_export.xlsx"
        )

        wb = openpyxl.Workbook()
        for i, table_data in enumerate(tables_to_export):
            title_suffix = (
                f" {table_index + 1}" if table_index is not None else f" {i + 1}"
            )
            ws = wb.active if i == 0 else wb.create_sheet()
            ws.title = f"Table{title_suffix}"
            _write_excel_sheet(ws, table_data)

        wb.save(output_path)

        return FileResponse(
            output_path,
            filename=output_filename,
            media_type=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
            background=None,  # file stays on disk; task_manager cleans it up later
        )

    except Exception as e:
        print(f"[ERROR] Failed to export table: {e}")
        return create_error_response(str(e))


@router.get("/export-all/{session_id}")
async def export_all_tables(session_id: str):
    """Export all tables from every processed page into a single Excel file."""
    try:
        if not is_safe_session_id(session_id):
            return create_error_response("Invalid session_id", status_code=400)

        temp_files = glob.glob(
            os.path.join(TEMP_DIR, f"temp_{session_id}_page_*.txt")
        )
        if not temp_files:
            return create_error_response(
                f"No page files found for session '{session_id}'. "
                "Make sure you have processed a document first.",
                status_code=404,
            )

        temp_files.sort(key=_page_num_from_path)

        wb = openpyxl.Workbook()
        first_sheet_used = False
        table_count = 0

        for filepath in temp_files:
            page_num = _page_num_from_path(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            parser = TableParser()
            parser.feed(content)

            for i, table_data in enumerate(parser.tables):
                table_count += 1
                sheet_title = f"P{page_num}_T{i + 1}"[:31]  # Excel 31-char limit

                if not first_sheet_used:
                    ws = wb.active
                    ws.title = sheet_title
                    first_sheet_used = True
                else:
                    ws = wb.create_sheet(title=sheet_title)

                _write_excel_sheet(ws, table_data)

        if table_count == 0:
            return create_error_response(
                "No HTML tables found in any of the session pages.", status_code=404
            )

        output_filename = f"session_{session_id}_all_tables.xlsx"
        output_path = os.path.join(TEMP_DIR, f"temp_{session_id}_all_export.xlsx")
        wb.save(output_path)

        return FileResponse(
            output_path,
            filename=output_filename,
            media_type=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
            background=None,
        )

    except Exception as e:
        print(f"[ERROR] Failed to export all tables: {e}")
        return create_error_response(str(e))
